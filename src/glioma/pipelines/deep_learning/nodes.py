import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader
from glioma.extras.datasets.dataset_spectral_analysis import DatasetForSpectralAnalysis
from glioma.extras.datasets.torch_model import TorchLocalModel
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def calculate_metrics(model, data_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for age, data, target in data_loader:
            output = model(data.to(device), age.to(device))
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.numpy())

    confusion_mat = confusion_matrix(all_labels, all_preds)

    # Calculate accuracy, sensitivity, and specificity
    accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    sensitivity = confusion_mat[1, 1] / (confusion_mat[1, 1] + confusion_mat[1, 0])
    specificity = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])

    return accuracy, sensitivity, specificity


class Net(nn.Module):
    def __init__(self,activation):
        """Initializing the 1D-CNN 

        Args:
            activation (nn.Functional): Activation function comes 
            from optuna optimization
        """        
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 14, 40)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(14, 7, 40)
        self.bn1=nn.BatchNorm1d(14)
        self.bn0=nn.BatchNorm1d(1)
        self.dropout=nn.Dropout(p=0.2)
        self.act=(activation)
        self.conv3 = nn.Conv1d(7, 7, 40,padding=0)
        self.fc1 = nn.Linear(455, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(11, 2)


    def forward(self, x,age):
        """Forward blocks of the model. Input data moves through from this function.

        Args:
            x (float): input MRS spectrum
            age (float): Age of patient

        Returns:
            [float]: output of the model
        """        
        x = self.pool(self.act(self.conv1(x)))  # -> n, 6, 14, 14
        x=self.dropout(x)
        x=self.bn1(x)
        x = self.pool(self.act(self.conv2(x)))  # -> n, 16, 5, 5
        x = self.pool(self.act(self.conv3(x)))  # -> n, 16, 5, 5      
        x=self.dropout(x)     
        x = x.view(-1, x.shape[1] *x.shape[2])        
        x = self.act(self.fc1(x)) 
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        age=(age).unsqueeze(1)/100
        x=torch.cat((x,age),dim=1)
        x = self.fc4(x)     # -> n, 10
                     
        return x
    

def train(log_interval, model, train_loader, optimizer, epoch):
    """This is for training loop of optuna

    Args:
        log_interval (int): Logging interval
        model (torch model): 1D-CNN model that we created by using model1DCNN.py 
        train_loader (dataloader): train data_loader for training 
        optimizer (torch.optimizer): you can select the optimizer
        epoch (int): Epoch number

    Returns:
        [float]: Returns training loss
    """    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    losses=[]
    for batch_idx, (age, data, target) in enumerate(train_loader):
        optimizer.zero_grad() 
        output = model(data.to(device),age.to(device)) # model prediction
        criterion = nn.CrossEntropyLoss() # loss function
        target = target.long().to(device)
        loss=criterion(output, target.to(device))# computing the loss  
        loss.backward() # backpropogation
        optimizer.step() # optimize
        losses.append(loss.item()) # listing loss
    return losses # returning the losses

    
def test(model, test_loader):
    """ Testing loop for optuna

    Args:
        model (torch.model): model to input
        test_loader (dataloader): dataloader for testing 

    Returns:
        accuracy [float]: accuracy
        acc1 [float]: accuracy of class1
        acc2 [float]: accuracy of class2
        losses [list]: loss list
    """
    
    model.eval()
    test_loss = 0 # initializing the required variables
    correct = 0
    num_classes=[0,0]
    num_samples=[0,0]
    losses=[]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for age, data, target in test_loader:
            output = model(data.to(device),age.to(device))
            criterion = nn.CrossEntropyLoss()
            test_loss=criterion(output, target.to(device)) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            losses.append(test_loss)
            for i,_ in enumerate(pred):
                if pred[i]==target[i]:
                    num_classes[target[i]]+=1
                num_samples[target[i]]+=1
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)  # Overall accuracy
    acc1 = 100. * num_classes[0] / len(test_loader.dataset)  # Class 0 accuracy
    acc2 = 100. * num_classes[1] / len(test_loader.dataset)  # Class 1 accuracy

    y_true = target.cpu().numpy()
    y_pred = pred.cpu().numpy()
    clf_report = classification_report(y_true, y_pred)
    
    return accuracy,acc1,acc2,losses, clf_report
 
def tester(checkpoint, X_test, Age_test, y_test, parameters):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model=model.to(device)
    model.eval()

    X_test_np = X_test.to_numpy()
    Age_test_np = Age_test.to_numpy()
    y_test_np = y_test.to_numpy()

    test_dataset = DatasetForSpectralAnalysis(X_test_np, Age_test_np, y_test_np)
    test_loader = DataLoader(test_dataset, batch_size=parameters['test_batch_size'], shuffle=False)
            
    # Calculate metrics on the testing set
    test_predictions, test_labels = get_predictions_and_labels(model, test_loader)

    # Generate classification report
    test_classification_report = classification_report(test_labels, test_predictions)

    print("Classification Report on Training Set:")
    print(test_classification_report)

    return test_classification_report


def get_predictions_and_labels(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for age, data, target in dataloader:
            output = model(data.to(device), age.to(device))
            predictions = output.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(target.numpy())

    return all_predictions, all_labels


def optimize_n_train_1dcnn(X_train, y_train, Age_train, parameters):
    def objective(trial):

        nonlocal best_booster, best_state_dict

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        params = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'n_splits': 5,
            'n_epochs' :trial.suggest_categorical('epochs',[50,60,100,500]),
            'seed': 0,
            'log_interval': 1,
            'save_model': False,
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            'momentum': trial.suggest_uniform('momentum', 0.1, 0.99),
            'optimizer': trial.suggest_categorical('optimizer', [optim.SGD, optim.RMSprop]),
            'activation': trial.suggest_categorical('activation', [F.relu, F.sigmoid, F.leaky_relu])
        }

        torch.manual_seed(params['seed'])
        kfold = KFold(n_splits=params['n_splits'], shuffle=True, random_state=params['seed'])

        X_train_np = np.array(X_train)#.to_numpy()
        y_train_np = y_train.to_numpy()
        Age_train_np = np.array(Age_train)

        best_accuracy = 0.0

        for fold, (train_indices, test_indices) in enumerate(kfold.split(X_train_np)):
            x_train, x_val = X_train_np[train_indices], X_train_np[test_indices]
            train_labels, val_labels = y_train_np[train_indices], y_train_np[test_indices]
            age_train, age_val = Age_train_np[train_indices], Age_train_np[test_indices]
            
            train_dataset = DatasetForSpectralAnalysis(x_train, age_train, train_labels)
            train_loader_fold = DataLoader(train_dataset, batch_size=parameters['train_batch_size'], shuffle=True)
            val_dataset = DatasetForSpectralAnalysis(x_val, age_val, val_labels)
            val_loader_fold = DataLoader(val_dataset, batch_size=parameters['test_batch_size'], shuffle=False)

            model = Net(params['activation']).to(device)
            optimizer = params['optimizer'](model.parameters(), lr=params['lr'])
            losses_train_fold = []
            losses_test_fold = []
            accuracies_fold=[]

            for epoch in range(1, params['n_epochs'] + 1):
                # print(len(losses_train_fold), epoch)
                loss_train = train(params['log_interval'], model, train_loader_fold, optimizer, epoch)
                test_accuracy,acc1,acc2,test_loss, clf_report_test = test(model, val_loader_fold) # function test
                losses_train_fold.append(loss_train[0])
                losses_test_fold.append(test_loss[0])
                accuracies_fold.append(test_accuracy)

                if test_loss[0] < best_booster:
                    best_booster = test_loss[0]
                    best_state_dict = model.state_dict()

                    checkpoint = {
                        'model': Net(params['activation']),
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epochs': epoch,
                        'activation': params['activation'],
                        'lr': params['lr'],
                        'batch_size': parameters['train_batch_size'],
                        'acc1': acc1,
                        'acc2': acc2,
                        'classification_report':clf_report_test,
                        'test_loss': losses_test_fold,
                        'train_loss': losses_train_fold,
                        'accuracies': accuracies_fold
                    }

                    model_saver = TorchLocalModel(filepath=f'data/06_models/1dcnn_model_TERT_{test_accuracy}.pth')
                    model_saver.save(checkpoint)
                    

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy

            print(f'Fold {fold + 1}/{params["n_splits"]}, Best Accuracy: {best_accuracy}%')

        return test_loss[0]
    
    # Initialize best_booster as a global variable
    best_booster = float('inf')
    best_state_dict = None

    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=50)

    return None
    
