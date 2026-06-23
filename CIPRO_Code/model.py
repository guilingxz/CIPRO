import os
import lap
import torch
import numpy
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn.utils
from prettytable import PrettyTable
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import plot_confusion_matrix


def save_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(filename):
    print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_accuracy(labels, prediction):    
    cm = confusion_matrix(labels, prediction)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):    
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]    
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy 

def calculate_confusion_matrix(labels, predictions):
    tn, fp, fn, tp = 0, 0, 0, 0

    for true_label, predicted_label in zip(labels, predictions):
        if true_label == 0 and predicted_label == 0:
            tn += 1
        elif true_label == 0 and predicted_label == 1:
            fp += 1
        elif true_label == 1 and predicted_label == 0:
            fn += 1
        elif true_label == 1 and predicted_label == 1:
            tp += 1

    return tn, fp, fn, tp

def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    tn, fp, fn, tp = calculate_confusion_matrix(labels, predictions)

    if tp + fp == 0:
        pr_array = 0
    else:
        pr_array = tp / (tp + fp)  # Precision

    if fp + tn == 0:
        fpr_array = 0
    else:
        fpr_array = fp / (fp + tn)

    if tp + fn == 0:
        fnr_array = 0
    else:
        fnr_array = fn / (tp + fn)

    if tp + fp + fn == 0:
        f1_array = 0
    else:
        f1_array = 2 * tp / (2 * tp + fp + fn)

    if tp + fn == 0:
        tpr_array = 0
    else:
        tpr_array = tp / (tp + fn)  # True Positive Rate (Recall)

    sum_array = fn + tp

    M_fpr = fpr_array
    M_fnr = fnr_array
    M_f1 = f1_array
    M_pr = float(pr_array)  # Convert to float
    M_re = float(tpr_array)  # Convert to float
    W_fpr = (fpr_array * sum_array) / sum_array
    W_fnr = (fnr_array * sum_array) / sum_array
    W_f1 = (f1_array * sum_array) / sum_array
    MCM = [[tn, fp], [fn, tp]]

    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1": format(M_f1 * 100, '.3f'),
        "M_pr": format(M_pr * 100, '.3f'),
        "M_re": format(M_re * 100, '.3f'),
        "W_f1": format(W_f1 * 100, '.3f'),
        "ACC": format(accuracy * 100, '.3f'),
        "MCM": None
    }

class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3,self.max_len,self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                if len(feature[j][i]) == 0:
                    # 如果是空列表，用全零向量替换
                    vectors[j][i] = np.zeros(128)
                else:
                    vectors[j][i] = feature[j][i]
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }

class TextCNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)           
        self.num_filters = 32                                        
        classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state

class CNN_Classifier():
    def __init__(self, max_len=100, n_classes=2, epochs=100, batch_size = 32, learning_rate = 0.001, \
                    result_save_path = "./result", item_num = 0, hidden_size = 128):
        self.model = TextCNN(hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        self.train_losses = []  # 存储每个 epoch 结束后的训练损失
        self.val_losses = []   # 存储每个 epoch 结束后的验证损失
        self.epoch_probabilities = []
        result_save_path = result_save_path + "/" if result_save_path[-1]!="/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(batch_size) + ".result"

    def preparation(self, X_train,  y_train, X_valid, y_valid):
        # create datasets
        self.train_set = TraditionalDataset(X_train, y_train, self.max_len, self.hidden_size)
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.hidden_size)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs,_  = self.model( vectors )
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward() # type: ignore
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # type: ignore
            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        self.train_losses.append(train_loss)
        self.epoch_probabilities.append(predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, _ = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = float(correct_predictions) / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        self.val_losses.append(val_loss)
        return val_loss, score_dict

    
    def train(self, model_save_path):
        best_val_loss = float('inf')  # 初始化最佳验证集损失为正无穷大
        best_model_state = None
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_pr', 'W_re', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_pr', 'W_re', 'W_f1', 'ACC'])
        last_epoch = -1
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                    "train_score": train_score, "val_score": val_score}
            save_data(self.result_save_path, learning_record_dict)
            last_epoch = epoch

            # 保存最优模型的 state_dict
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_save_path = f"{model_save_path}_best_model.pth"
                torch.save(self.model.state_dict(), best_model_save_path)
                print(f"Best model saved at {best_model_save_path}")
        
        # 在训练结束后保存最后一代模型
        last_model_save_path = f"{model_save_path}_last_model.pth"
        torch.save(self.model.state_dict(), last_model_save_path)
        print(f"Last model saved at {last_model_save_path}")
        
        #fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)

        # # 绘制 ROC 曲线
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend()

        # # 保存 ROC 曲线图
        # roc_curve_save_path = f"{model_save_path}_roc_curve.png"
        # plt.savefig(roc_curve_save_path)
        # print(f"ROC curve plot saved at {roc_curve_save_path}")

        # 保存混淆矩阵
        # confusion_matrix_save_path = f"{model_save_path}_confusion_matrix.png"
        # self.plot_confusion_matrix(confusion_matrix_save_path)
        # print(f"Confusion matrix saved at {confusion_matrix_save_path}")

        # 保存训练过程图
        training_plot_save_path = f"{model_save_path}_training_plot.png"
        self.plot_training_process(training_plot_save_path)
        print(f"Training process plot saved at {training_plot_save_path}")
    
    # def plot_confusion_matrix(self, save_path):
    #     # 绘制混淆矩阵
    #     plt.figure(figsize=(8, 6))
    #     sns.set(font_scale=1.2)
    #     plot_confusion_matrix(self.model, self.valid_loader, cmap=plt.cm.Blues)
    #     plt.title("Confusion Matrix")
    #     plt.savefig(save_path)
    #     plt.close()

    def plot_training_process(self, save_path):
        # 绘制训练过程图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Process")
        plt.savefig(save_path)
        plt.close()
    def test(self, test_loader):
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, _ = self.model(vectors)
                
                # Define a new loss function for testing
                test_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                losses.append(test_loss.item())

        test_acc = float(correct_predictions) / len(test_loader.dataset)
        print("Test Accuracy: ", test_acc)

        # 计算并输出测试集的性能指标，可以使用get_MCM_score或其他性能指标函数
        test_score = get_MCM_score(label, pre)
        print("Test Performance:")
        print(test_score)

        test_loss = np.mean(losses)
        return test_loss, test_score
    def load_model_state(self, model_path):
        self.load_state_dict(torch.load(model_path))