'''Maximum Mean Discrepancy (MMD):
MMD is a kernel-based distance metric used to compare two sets of samples or probability distributions. 
It quantifies the distance between the means of the distributions in a high-dimensional feature space induced by a kernel function. 
The MMD metric can be used to determine if two sets of samples are drawn from the same distribution or to compare the similarity of
two probability distributions. MMD has been commonly used in various machine learning tasks, 
such as domain adaptation, generative model evaluation, and two-sample testing.'''
import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home+'/projects/amt_approx_simplex')
import torch
from model_zoo import mnist_net, mnist_net_f, mnist_net_g
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.datasets import EMNIST
import numpy as np

num_samples = 500

def train_bayesian(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output,dim=1), target)
        loss.backward()
        optimizer.step()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output,dim=1), target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def polynomial(x, y, B, c, d):
    # calculate mmd with polynomial kernel in batch

    xx, yy, zz = torch.bmm(x, x.permute(0,2,1)), \
                 torch.bmm(y, y.permute(0,2,1)), \
                 torch.bmm(x, y.permute(0,2,1))

    K = torch.pow((xx + c), d)
    L = torch.pow((yy + c), d)
    P = torch.pow((zz + c), d)

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K, [1, 2]) + torch.sum(L, [1, 2])) - gamma * torch.sum(P, [1, 2])

def batch_mmd(x, y, B, alpha):
    # calculate mmd with RBF kernel in batch
    xx, yy, zz = torch.bmm(x, x.permute(0,2,1)), \
                 torch.bmm(y, y.permute(0,2,1)), \
                 torch.bmm(x, y.permute(0,2,1))

    rx = (xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(xx))
    ry = (yy.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(yy))

    K = torch.exp(- alpha * (rx.permute(0,2,1) + rx - 2 * xx))
    L = torch.exp(- alpha * (ry.permute(0,2,1) + ry - 2 * yy))
    P = torch.exp(- alpha * (rx.permute(0,2,1) + ry - 2 * zz))

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K, [1,2]) + torch.sum(L, [1,2])) - gamma * torch.sum(P, [1,2])
    
'''Here is the student model that approximates the teacher distribution by using the less expensive Dirichlet distribution.
The reason the student model is not intractable and can be more computationally efficient compared to the teacher model is because
it uses the Dirichlet distribution as its parametric distribution. In the OPU (One-Pass Uncertainty) framework with the student model, 
there is no need for additional sampling during inference. The student model directly provides class probabilities as its 
output, and these class probabilities can be used to obtain the level of uncertainty in its predictions.'''
def train_approx(args, fmodel, gmodel, device, approx_loader, f_optimizer, g_optimizer, output_samples, epoch):
    gmodel.train()
    fmodel.train()
    for batch_idx, (data, target) in enumerate(approx_loader):
        data, target = data.to(device), target.to(device)
        f_optimizer.zero_grad()

        with torch.no_grad():
            # To be consistant with KL, the exp() function is changed to softplus,
            # i.e., alpha0 = softplus(g).
            # Note that, for mmd, the exp() function can be used directly for faster convergence,
            # without tuning hyper-parameters.
            g_out = F.softplus(gmodel(data))
            output = output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
                device).clamp(0.0001, 0.9999)

        f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)


        '''Instead of directly working with the entire Dirichlet distribution, the code is taking samples from the Dirichlet 
        distribution to calculate the loss. This approach is known as Monte Carlo sampling.
        The Dirichlet distribution represents a family of probability distributions over a simplex (a geometric shape 
        that generalizes a triangle to higher dimensions) with K dimensions (K being the number of classes or categories). 
        Each sample from the Dirichlet distribution corresponds to a probability vector (π) of length K, where each component
        represents the probability of an item belonging to a specific class.'''
        
        s1 = torch.distributions.Dirichlet(pi).rsample((num_samples,)).permute(1,0,2)

        #MMD usage:
        '''The MMD (Maximum Mean Discrepancy) and polynomial functions are used to compare the distribution of class 
        probabilities from the model's output (output) with the sampled distribution (s1). The loss function aims to minimize 
        the discrepancy between these distributions.'''
        loss = (batch_mmd(output, s1, num_samples, 1e5)
                + 0.5*polynomial(output, s1, num_samples, 1, 3)
                + 0.5*polynomial(output, s1, num_samples, 1, 4)).mean()

        loss.backward()
        f_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

        g_optimizer.zero_grad()

        g_out = F.softplus(gmodel(data))

        with torch.no_grad():
            output = output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
            device).clamp(0.0001, 0.9999)

        with torch.no_grad():
            f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)
        s1 = torch.distributions.Dirichlet(pi).rsample((num_samples,)).permute(1,0,2)

        loss = (batch_mmd(output, s1, num_samples, 1e5)
                + 0.5*polynomial(output, s1, num_samples, 1, 3)
                + 0.5*polynomial(output, s1, num_samples, 1, 4)).mean()

        loss.backward()
        g_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

#EVALUATION CODE: Only Measure uncertainty to distinguish between ID and ODD, does not already know labels, evaluates on
#test set and OOD set and sees through AUROC and AUPR whether it discriminates properly
'''The provided code is evaluating the performance of a model on out-of-distribution (OOD) data and assessing its ability to 
distinguish between in-distribution (ID) and OOD samples using various uncertainty metrics. The metrics such as AUROC 
(Area Under the Receiver Operating Characteristic curve) and AUPR (Area Under the Precision-Recall curve) are used to 
quantify how well the model can distinguish between the two categories:

1. The code concatenates the values of uncertainty metrics (`entros_origin_2`, `fentros_approx_2`, etc.) obtained for 
out-of-distribution data (`ood_loader`) with the metrics obtained for in-distribution data (`test_loader`).

2. A binary label array `ood` is created, where zeros represent in-distribution samples, and ones represent out-of-distribution samples.

3. The AUROC and AUPR scores are calculated for each uncertainty metric using the `roc_auc_score` and `average_precision_score` 
functions. These metrics quantify the model's ability to differentiate between in-distribution and out-of-distribution samples based 
on the uncertainty values.

4. AUROC Calculation: The AUROC score is calculated by plotting the Receiver Operating Characteristic curve. The ROC curve is 
created by varying the threshold for classifying samples as positive or negative. For each threshold, the true positive rate 
(TPR) and false positive rate (FPR) are calculated. The AUROC score is then calculated as the area under this curve. 
It represents the model's ability to discriminate between the two classes across various threshold values.
AUPR Calculation: The AUPR score is calculated by plotting the Precision-Recall curve. The precision and recall are calculated for 
different threshold values. Precision is the ratio of true positives to the total predicted positives, while recall is the ratio of 
true positives to the total actual positives. The AUPR score is the area under this curve, representing how well the model ranks the 
positive class samples with respect to the negative class samples.

5. The calculated AUROC and AUPR scores are printed for each uncertainty metric, providing insights into how well the model performs 
at identifying OOD data. Higher AUROC and AUPR scores indicate better discrimination between in-distribution and OOD data.
The code essentially evaluates the model's robustness against OOD samples by analyzing how well it captures uncertainty patterns
that differentiate between the two types of data. If the AUROC and AUPR scores are high, it suggests that the model's uncertainty
metrics are effective in distinguishing between in-distribution and OOD samples'''


def eval_approx(args,  smean, sconc, device, test_loader,
                ood_loader, teacher_test_samples, teacher_ood_samples):

    smean.eval()
    sconc.eval()
    miscls_origin = []
    miscls_approx = []
    entros_origin_1 = []
    fentros_approx_1 = []
    entros_approx_1 = []
    entros_origin_2 = []
    fentros_approx_2 = []
    entros_approx_2 = []
    maxp_origin_1 = []
    maxp_approx_1 = []
    maxp_origin_2 = []
    maxp_approx_2 = []
    gvalue_approx_1 = []
    gvalue_approx_2 = []
    batch_idx = 0
    #This is evaluation on test loader
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1)
            pi_q = f_out.mul(g_out)

            samples_p_pi = teacher_test_samples[batch_idx*test_loader.batch_size: (batch_idx+1)*test_loader.batch_size].to(device)

            avg_origin_output = torch.mean(samples_p_pi, dim=1)

            pi_p_avg_batch = avg_origin_output
            origin_result = torch.argmax(pi_p_avg_batch, dim=1)
            approx_result = torch.argmax(pi_q, dim=1)

            miscls_approx.append((1.-(approx_result == target).float()).cpu().numpy())
            miscls_origin.append((1.-(origin_result == target).float()).cpu().numpy())

            entro_origin = (-torch.bmm(pi_p_avg_batch.view(data.shape[0], 1, -1),
                                      torch.log(pi_p_avg_batch.view(data.shape[0], -1, 1)))).view(-1)

            fentro_approx = (-torch.bmm(f_out.view( data.shape[0], 1, -1),
                                      torch.log(f_out.view(data.shape[0], -1, 1)))).view(-1)

            alpha = pi_q
            alpha0 = alpha.sum(1)

            entro_approx = torch.lgamma(alpha).sum(1) \
                           - torch.lgamma(alpha0) \
                           + (alpha0 - 47).mul(torch.digamma(alpha0)) \
                           - ((alpha - 1 ).mul(torch.digamma(alpha))).sum(1)

            entros_origin_1.append(entro_origin.cpu().numpy())
            fentros_approx_1.append(fentro_approx.cpu().numpy())
            entros_approx_1.append(entro_approx.cpu().numpy())

            maxp_origin = 1./torch.max(pi_p_avg_batch, dim=1)[0]
            maxp_approx = 1./torch.max(f_out, dim=1)[0]

            maxp_origin_1.append(maxp_origin.cpu().numpy())
            maxp_approx_1.append(maxp_approx.cpu().numpy())
            # gvalue_origin_1.append(g_out.cpu().numpy())
            gvalue_approx_1.append(1./g_out.cpu().numpy())
            batch_idx += 1

    miscls_approx = np.concatenate(miscls_approx)
    miscls_origin = np.concatenate(miscls_origin)
    entros_origin_1 = np.concatenate(entros_origin_1)
    fentros_approx_1 = np.concatenate(fentros_approx_1)
    maxp_origin_1 = np.concatenate(maxp_origin_1)
    maxp_approx_1 = np.concatenate(maxp_approx_1)
    gvalue_approx_1 = np.concatenate(gvalue_approx_1)
    correct_approx = np.sum(1-miscls_approx)
    correct_ensemble = np.sum(1-miscls_origin)

    ## Calculate and print the AUROC (Area Under the Receiver Operating Characteristic) scores for different metrics
    print("AUROC (entros_origin_1): ", roc_auc_score(miscls_origin, entros_origin_1))
    print("AUROC (fentros_approx_1): ", roc_auc_score(miscls_approx, fentros_approx_1))
    print("AUROC (maxp_approx_1):   ", roc_auc_score(miscls_approx, maxp_approx_1))
    print("AUROC (maxp_origin_1):   ", roc_auc_score(miscls_origin, maxp_origin_1))
    print("AUROC (gvalue_approx_1): ", roc_auc_score(miscls_approx, gvalue_approx_1))

    # Calculate and print the AUPR (Area Under the Precision-Recall) scores for different metrics
    print("AUPR  (entros_origin_1): ", average_precision_score(miscls_origin, entros_origin_1))
    print("AUPR  (fentros_approx_1): ", average_precision_score(miscls_approx, fentros_approx_1))
    print("AUPR  (maxp_approx_1):   ", average_precision_score(miscls_approx, maxp_approx_1))
    print("AUPR  (maxp_origin_1):   ", average_precision_score(miscls_origin, maxp_origin_1))
    print("AUPR  (gvalue_approx_1): ", average_precision_score(miscls_approx, gvalue_approx_1))
    print('approx ACC :', correct_approx / (len(test_loader.dataset)))
    print('ensemble ACC :', correct_ensemble / (len(test_loader.dataset)))

    #This is the evaluation on the OOD loader
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ood_loader):
            data, target = data.to(device), target.to(device)
            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1)
            pi_q = f_out.mul(g_out)

            samples_p_pi = teacher_ood_samples[batch_idx*ood_loader.batch_size: (batch_idx+1)*ood_loader.batch_size].to(device)

            avg_origin_output = torch.mean(samples_p_pi, dim=1)

            pi_p_avg_batch = avg_origin_output
            entro_origin = (-torch.bmm(pi_p_avg_batch.view(data.shape[0], 1, -1),
                                      torch.log(pi_p_avg_batch.view(data.shape[0], -1, 1)))).view(-1)

            fentro_approx = (-torch.bmm(f_out.view( data.shape[0], 1, -1),
                                      torch.log(f_out.view(data.shape[0], -1, 1)))).view(-1)

            entros_origin_2.append(entro_origin.cpu().numpy())
            fentros_approx_2.append(fentro_approx.cpu().numpy())

            alpha = pi_q
            alpha0 = alpha.sum(1)

            entro_approx = torch.lgamma(alpha).sum(1) \
                           - torch.lgamma(alpha0) \
                           + (alpha0 - 47).mul(torch.digamma(alpha0)) \
                           - ((alpha - 1 ).mul(torch.digamma(alpha))).sum(1)

            entros_approx_2.append(entro_approx.cpu().numpy())

            maxp_origin = 1./torch.max(pi_p_avg_batch, dim=1)[0]
            maxp_approx = 1./torch.max(f_out, dim=1)[0]

            maxp_origin_2.append(maxp_origin.cpu().numpy())
            maxp_approx_2.append(maxp_approx.cpu().numpy())
            gvalue_approx_2.append(1./g_out.cpu().numpy())
            batch_idx += 1

        entros_origin_2 = np.concatenate(entros_origin_2)
        fentros_approx_2 = np.concatenate(fentros_approx_2)
        maxp_origin_2 = np.concatenate(maxp_origin_2)
        maxp_approx_2 = np.concatenate(maxp_approx_2)
        gvalue_approx_2 = np.concatenate(gvalue_approx_2)

        fentros_approx = np.concatenate([fentros_approx_1, fentros_approx_2])
        entros_origin = np.concatenate([entros_origin_1 , entros_origin_2])
        maxp_approx = np.concatenate([maxp_approx_1 , maxp_approx_2])
        maxp_origin = np.concatenate([maxp_origin_1 , maxp_origin_2])
        gvalue_approx = np.concatenate([gvalue_approx_1 , gvalue_approx_2])
        ood = np.concatenate([np.zeros(test_loader.dataset.__len__()),
                              np.ones(ood_loader.dataset.__len__())])

        print("-----------------------")
        print("AUROC (entros_origin): ", roc_auc_score(ood, entros_origin))
        print("AUROC (fentros_approx): ", roc_auc_score(ood, fentros_approx))
        print("AUROC (maxp_approx):   ", roc_auc_score(ood, maxp_approx))
        print("AUROC (maxp_origin):   ", roc_auc_score(ood, maxp_origin))
        print("AUROC (gvalue_approx): ", roc_auc_score(ood, gvalue_approx))
        print("AUPR  (entros_origin): ", average_precision_score(ood, entros_origin))
        print("AUPR  (fentros_approx): ", average_precision_score(ood, fentros_approx))
        print("AUPR  (maxp_approx):   ", average_precision_score(ood, maxp_approx))
        print("AUPR  (maxp_origin):   ", average_precision_score(ood, maxp_origin))
        print("AUPR  (gvalue_approx): ", average_precision_score(ood, gvalue_approx))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='run approximation to LeNet on Mnist')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--approx-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to approx (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dropout-rate', type=float, default=0.5, metavar='p_drop',
                        help='dropout rate')
    parser.add_argument('--S', type=int, default=100, metavar='N',
                        help='number of posterior samples from the Bayesian model')
    parser.add_argument('--model-path', type=str, default='../saved_models/emnist_mcdp/',
                        help='number of posterior samples from the Bayesian model')
    parser.add_argument('--save-approx-model', type=int, default=0, metavar='N',
                        help='save approx model or not? default not')
    parser.add_argument('--from-approx-model', type=int, default=1, metavar='N',
                        help='if our model is loaded or trained')
    parser.add_argument('--test-ood-from-disk', type=int, default=1,
                        help='generate test samples or load from disk')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}

    tr_data = EMNIST('../data', split='balanced', train=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    te_data = EMNIST('../data', split='balanced', train=False, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    ood_data = datasets.Omniglot('../data', download=True,  transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))

    train_loader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        te_data,
        batch_size=args.batch_size, shuffle=False,  **kwargs)

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = mnist_net().to(device)

    model.load_state_dict(torch.load(args.model_path + 'mcdp-emnist.pt'))

    test(args, model, device, test_loader)

    if args.from_approx_model == 0:
        output_samples = torch.load(args.model_path + 'emnist-mcdp-samples.pt')

    # --------------- training approx ---------

    print('approximating ...')
    fmodel = mnist_net_f().to(device)
    gmodel = mnist_net_g().to(device)

    if args.from_approx_model == 0:
        g_optimizer = optim.SGD(gmodel.parameters(), lr=args.lr, momentum=args.momentum)
        f_optimizer = optim.SGD(fmodel.parameters(), lr=args.lr, momentum=args.momentum)
        best_acc = 0
        for epoch in range(1, args.approx_epochs + 1):
            train_approx(args, fmodel, gmodel, device, train_loader, f_optimizer, g_optimizer, output_samples, epoch)
            acc = test(args, fmodel, device, test_loader)
            if acc > best_acc:
                torch.save(fmodel.state_dict(), args.model_path + 'mcdp-emnist-mean-mmd.pt')
                torch.save(gmodel.state_dict(), args.model_path + 'mcdp-emnist-conc-mmd.pt')
                best_acc = acc

    else:
        fmodel.load_state_dict(torch.load(args.model_path + 'mcdp-emnist-mean-mmd.pt'))
        gmodel.load_state_dict(torch.load(args.model_path + 'mcdp-emnist-conc-mmd.pt'))

    print('generating teacher particles for testing&ood data ...')
    # generate particles for test and ood dataset
    model.train()
    if args.test_ood_from_disk == 1:
        teacher_test_samples = torch.load(args.model_path + 'emnist-mcdp-test-samples.pt')
    else:
        with torch.no_grad():
            # obtain ensemble outputs
            all_samples = []
            for i in range(500):
                samples_a_round = []
                for data, target in test_loader:
                    data = data.to(device)
                    output = F.softmax(model(data), dim=X)
                    samples_a_round.append(output)
                samples_a_round = torch.cat(samples_a_round).cpu()
                all_samples.append(samples_a_round)
            teacher_test_samples = torch.stack(all_samples).permute(1,0,2)

            torch.save(all_samples, args.model_path + 'emnist-mcdp-test-samples.pt')

    if args.test_ood_from_disk == 1:
        teacher_ood_samples = torch.load(args.model_path + 'omniglot-mcdp-ood-samples-trd-emnist.pt')
    else:
        with torch.no_grad():
            # obtain ensemble outputs
            all_samples = []
            for i in range(500):
                samples_a_round = []
                for data, target in ood_loader:
                    data = data.to(device)
                    output = F.softmax(model(data), dim=X)
                    samples_a_round.append(output)
                samples_a_round = torch.cat(samples_a_round).cpu()
                all_samples.append(samples_a_round)
            teacher_ood_samples = torch.stack(all_samples).permute(1,0,2)

            torch.save(all_samples, args.model_path + 'omniglot-mcdp-ood-samples-trd-emnist.pt')

    eval_approx(args, fmodel, gmodel, device, test_loader, ood_loader, teacher_test_samples, teacher_ood_samples)

if __name__ == '__main__':
    main()
