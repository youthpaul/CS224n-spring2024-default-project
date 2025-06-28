#!/usr/bin/env python3

'''
Multitask BERT evaluation functions.

When training your multitask model, you will find it useful to call
model_eval_multitask to evaluate your model on the 3 tasks' dev sets.
'''

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import stats


TQDM_DISABLE = False


# Evaluate multitask model on sentiment analysis
def model_eval_sa(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'sa-eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, y_pred, y_true

# Evaluate paraphrase detection
def model_eval_pd(sts_dataloader, model, device):
    with torch.no_grad():
        pd_y_true = []
        pd_y_pred = []
        pd_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'para-eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            pd_y_pred.extend(y_hat)
            pd_y_true.extend(b_labels)
            pd_sent_ids.extend(b_sent_ids)
        pd_acc = accuracy_score(pd_y_true, pd_y_pred)

        return pd_acc, pd_y_pred, pd_y_true


# Evaluate semantic textual similarity.
def model_eval_sts(sts_dataloader, model, device):
    with torch.no_grad():
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'sts-eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]

        return sts_corr, sts_y_pred, sts_y_true


# Evaluate multitask model on dev sets.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Visualize for Sentiment Analysis
        sa_acc, sa_y_pred, sa_y_true = model_eval_sa(sentiment_dataloader, model, device)

        cm = confusion_matrix(sa_y_true, sa_y_pred, labels=[0,1,2,3,4])

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['0','1','2','3','4'],
                    yticklabels=['0','1','2','3','4'])
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('./img/sentiment_cm.png', dpi=300)
        plt.close()


        # Visualization for Paraphrase Detection
        pd_acc, para_y_pred, para_y_true = model_eval_pd(paraphrase_dataloader, model, device)

        plt.figure(figsize=(6,5))
        cm = confusion_matrix(para_y_true, para_y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0','1'],
                    yticklabels=['0','1'])
        plt.title('Paraphrase Detection Performance')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('./img/paraphrase_cm.png', dpi=300, bbox_inches='tight')
        plt.close()


        # Scatter for STS
        sts_corr, sts_y_pred, sts_y_true = model_eval_sts(sts_dataloader, model, device)

        plt.figure(figsize=(8, 6))
        plt.scatter(sts_y_pred, sts_y_true, alpha=0.4, s=15, color='green')
        
        pearson_corr, _ = stats.pearsonr(sts_y_pred, sts_y_true)
        # plt.text(0.5, 4.7, f'Pearson: {pearson_corr:.3f}', 
        #         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Semantic Textual Similarity', fontsize=14)
        plt.xlabel('Predictions', fontsize=12)
        plt.ylabel('Similarity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        # plt.xlim(0, 5)
        # plt.ylim(0, 5)
        plt.savefig('./img/sts_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        return sa_acc, pd_acc, sts_corr



# Evaluate multitask model on test sets.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        # Evaluate paraphrase detection.
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)

        # Evaluate semantic textual similarity.
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)

        return (sst_y_pred, sst_sent_ids,
                para_y_pred, para_sent_ids,
                sts_y_pred, sts_sent_ids)
