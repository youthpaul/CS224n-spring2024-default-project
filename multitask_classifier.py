'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_multitask, model_eval_test_multitask
from evaluation import model_eval_sa, model_eval_pd, model_eval_sts

from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig
from peft import PeftConfig, PeftModel

from tokenizer import BertTokenizer

from pcgrad import PCGrad
import itertools
import os

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

lora_config = LoraConfig(
    # task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'attention_dense',
                    'interm_dense', 'out_dense']
)


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('/root/autodl-fs/bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.sentiment_layer = nn.Linear(config.hidden_size, 5)
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.paraphrase_layer = nn.Linear(config.hidden_size * 2, 1)
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.scale = nn.Linear(1, 1, bias=False)
        self.similarity_layer = nn.Linear(config.hidden_size, 1)
        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-fs/bert-base-uncased')


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        bert_output = self.bert(input_ids, attention_mask)
        return bert_output["pooler_output"]


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        bert_output = self.forward(input_ids, attention_mask)
        outputs = self.sentiment_dropout(bert_output)
        logits = self.sentiment_layer(outputs)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        bert_output_2 = self.forward(input_ids_2, attention_mask_2)

        concatenated_output = torch.cat([bert_output_1, bert_output_2], dim=-1)
        outputs = self.paraphrase_dropout(concatenated_output)
        outputs = self.paraphrase_layer(outputs)

        return outputs


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        # Cross Encoding
        # 1. concate tow sentences
        sep_id = self.tokenizer.sep_token_id
        input_ids = torch.cat([
            input_ids_1, 
            input_ids_2[:, 1:]
        ], dim=1)

        
        # 2. set 1 to the middle [SEP] mask
        attention_mask = torch.cat([
            attention_mask_1[:, :-1],
            torch.ones_like(attention_mask_1[:, :1]),  # [SEP] potision mask set to 1
            attention_mask_2[:, 1:]
        ], dim=1)

        
        # 3. final single sentence
        cls_output = self.forward(input_ids, attention_mask)

        # 4. using the [CLS] token to predict
        outputs = self.similarity_dropout(cls_output)
        return self.similarity_layer(outputs).squeeze()


        # Baseline
        # bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        # bert_output_2 = self.forward(input_ids_2, attention_mask_2)

        # concatenated_output = torch.cat([bert_output_1, bert_output_2], dim=-1)
        # outputs = self.similarity_dropout(concatenated_output)
        # return self.similarity_layer(outputs).squeeze()

        # using cosine similarity for STS
        # cosine_sim = F.cosine_similarity(bert_output_1, bert_output_2, dim=-1)  # [batch]

        # scalling
        # scaled_sim = self.scale(cosine_sim.unsqueeze(1)) * 2.5 + 2.5
        # return scaled_sim.squeeze()




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        # 'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")



def save_config(args, config, filepath):
    save_info = {
        'args': args,
        'model_config': config,
    }

    torch.save(save_info, filepath)
    print(f"save the config to {filepath}")


def check_gradients(model):
    """checking the gradient"""
    has_gradient = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            if grad_mean > 1e-7:  # 文档建议的阈值
                print(f" 梯度正常: {name} | 均值: {grad_mean:.3e}")
                has_gradient = True
    if not has_gradient:
        print("\n\n===>  检测到梯度消失! \n\n")


def train_sa(model, optimizer, args, epoch, train_dataloader, device):
    # Trainging for Sentiment analysis
    model.train()
    loss_sa = []
    for batch in tqdm(train_dataloader, desc=f'sa-train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                    batch['attention_mask'], batch['labels'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        # optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

        loss_sa.append(loss)

        # loss.backward()
        # check_gradients(model)
        # optimizer.step()
    return loss_sa




def train_pd(model, optimizer, args, epoch, train_dataloader, device):
    # Trainging for Paraphrase detection
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    loss_pd = []
    for batch in tqdm(train_dataloader, desc=f'para-train-{epoch}', disable=TQDM_DISABLE):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                            batch['token_ids_2'], batch['attention_mask_2'],
                                                            batch['labels'])

        b_ids_1 = b_ids_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device)

        # optimizer.zero_grad()
        logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float().view(-1), reduction='mean')

        loss_pd.append(loss)

        # loss.backward()
        # check_gradients(model)
        # optimizer.step()
    return loss_pd



def train_sts(model, optimizer, args, epoch, train_dataloader, device):
    # using regression loss to train
    model.train()
    loss_sts = []
    for batch in tqdm(train_dataloader, desc=f'sts-train-{epoch}', disable=TQDM_DISABLE):
        # data loading
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
            batch['token_ids_1'], batch['attention_mask_1'],
            batch['token_ids_2'], batch['attention_mask_2'],
            batch['labels']
        )
        
        b_ids_1 = b_ids_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device).float()  # to float
        
        # optimizer.zero_grad()
        
        # using Cross Encoding
        predictions = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        
        loss = F.mse_loss(predictions, b_labels, reduction='sum') / args.batch_size

        loss_sts.append(loss)
        
        # loss.backward()
        # check_gradients(model)
        # optimizer.step()
    return loss_sts
        

def train_multi(model, sa_batch, pd_batch, sts_batch, args, device):
    model.train()
    sa_b_ids, sa_b_mask, sa_b_labels = (sa_batch['token_ids'],
                            sa_batch['attention_mask'], sa_batch['labels'])
    pd_b_ids_1, pd_b_mask_1, pd_b_ids_2, pd_b_mask_2, pd_b_labels = (pd_batch['token_ids_1'], 
                                                    pd_batch['attention_mask_1'],
                                                    pd_batch['token_ids_2'], 
                                                    pd_batch['attention_mask_2'],
                                                    pd_batch['labels'])
    sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2, sts_b_labels = (
            sts_batch['token_ids_1'], sts_batch['attention_mask_1'],
            sts_batch['token_ids_2'], sts_batch['attention_mask_2'],
            sts_batch['labels'])
    
    sa_b_ids = sa_b_ids.to(device)
    sa_b_mask = sa_b_mask.to(device)
    sa_b_labels = sa_b_labels.to(device)

    pd_b_ids_1 = pd_b_ids_1.to(device)
    pd_b_ids_2 = pd_b_ids_2.to(device)
    pd_b_mask_1 = pd_b_mask_1.to(device)
    pd_b_mask_2 = pd_b_mask_2.to(device)
    pd_b_labels = pd_b_labels.to(device)

    sts_b_ids_1 = sts_b_ids_1.to(device)
    sts_b_ids_2 = sts_b_ids_2.to(device)
    sts_b_mask_1 = sts_b_mask_1.to(device)
    sts_b_mask_2 = sts_b_mask_2.to(device)
    sts_b_labels = sts_b_labels.to(device).float()  # to float

    # sentiment analysis
    sa_logits = model.predict_sentiment(sa_b_ids, sa_b_mask)
    sa_loss = F.cross_entropy(sa_logits, sa_b_labels.view(-1), reduction='sum') / args.batch_size

    # paraphrase detection
    pd_logits = model.predict_paraphrase(pd_b_ids_1, pd_b_mask_1, pd_b_ids_2, pd_b_mask_2)
    pd_loss = F.binary_cross_entropy_with_logits(pd_logits.view(-1), pd_b_labels.float().view(-1), reduction='mean')

    # sts
    sts_predictions = model.predict_similarity(sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2)
    sts_loss = F.mse_loss(sts_predictions, sts_b_labels, reduction='sum') / args.batch_size

    return sa_loss, pd_loss, sts_loss

from itertools import islice

def truncate_dataloader(dataloader, max_batches=10):
    """
    truncate the dataloader to speed debug
    """
    class TruncatedLoader:
        def __init__(self, original_loader, max_batches):
            self.original_loader = original_loader
            self.max_batches = max_batches
            self.batch_size = original_loader.batch_size
            self.collate_fn = original_loader.collate_fn
            
        def __iter__(self):
            # 使用islice截断迭代器
            return islice(self.original_loader, self.max_batches)
            
        def __len__(self):
            return min(self.max_batches, len(self.original_loader))
    
    return TruncatedLoader(dataloader, max_batches)


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='dev')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sa_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sa_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    pd_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_train_data.collate_fn)
    pd_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # See all layers' name
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(f"- {name}")

    # Apply LoRA
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = PCGrad(optimizer) # apply PCGrad
    best_dev_acc = 0
    
    if args.debug:
        N = args.truncate_size
        pd_train_dataloader = truncate_dataloader(pd_train_dataloader, N)
        sts_train_dataloader = truncate_dataloader(sts_train_dataloader, N)
        sa_train_dataloader = truncate_dataloader(sa_train_dataloader, N)
        pd_dev_dataloader = truncate_dataloader(pd_dev_dataloader, N)
        sts_dev_dataloader = truncate_dataloader(sts_dev_dataloader, N)
        sa_dev_dataloader = truncate_dataloader(sa_dev_dataloader, N)

    # count the max length
    sa_batch_count = len(sa_train_dataloader)
    pd_batch_count = len(pd_train_dataloader)
    sts_batch_count = len(sts_train_dataloader)
    max_batches = max(sa_batch_count, pd_batch_count, sts_batch_count)

    # create cyclic iterater to sample
    sa_cyclic = itertools.cycle(sa_train_dataloader)
    pd_cyclic = itertools.cycle(pd_train_dataloader)
    sts_cyclic = itertools.cycle(sts_train_dataloader)

    patience = 0 # number of model dev acc stay unchange

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        total_loss = 0.0
        
        # train for PCGrad
        with tqdm(total=max_batches, desc=f'Training of Epoch {epoch}:', 
                bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
            
            # interleaved training
            for step in range(max_batches):
                sa_batch = next(sa_cyclic)
                pd_batch = next(pd_cyclic)
                sts_batch = next(sts_cyclic)
                
                # compute the loss of three tasks
                loss_sa, loss_pd, loss_sts = train_multi(model,sa_batch, pd_batch,
                                                        sts_batch, args, device)
                
                # PCGrad
                optimizer.zero_grad()
                optimizer.pc_backward([loss_sa, loss_pd, loss_sts])
                optimizer.step()

                pbar.update(1)
                
        # loss_sa = train_sa(model, optimizer, args, epoch, sa_train_dataloader,
        #                          device)
        # loss_pd = train_pd(model, optimizer, args, epoch, pd_train_dataloader,
        #                           device)
        # loss_sts = train_sts(model, optimizer, args, epoch, sts_train_dataloader, device)

        # PCGrad
        # optimizer.zero_grad()
        # optimizer.pc_backward(loss_sa + loss_pd + loss_sts)
        # optimizer.step()

        # loss = loss_sa + loss_pd + loss_sts
        # loss = [el.to('cpu') for el in loss]
        # optimizer.minimize(loss)

        # evaluate for train
        train_sa_acc, *_ = model_eval_sa(sa_train_dataloader, model, device)
        train_pd_acc, *_ = model_eval_pd(pd_train_dataloader, model, device)
        train_sts_corr, *_ = model_eval_sts(sts_train_dataloader, model, device)

        print(f'\nepoch{epoch} trainging loss:')
        print(f'Sentiment Analysis accuracy: {train_sa_acc:.3f}')
        print(f'Paraphrase Detection accuracy: {train_pd_acc:.3f}')
        print(f'Semantic Textual Similarity correlation: {train_sts_corr:.3f}\n')

        # evaluate for dev
        dev_sa_acc, *_ = model_eval_sa(sa_dev_dataloader, model, device)
        dev_pd_acc, *_ = model_eval_pd(pd_dev_dataloader, model, device)
        dev_sts_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device)

        print(f'\nepoch{epoch} dev loss:')
        print(f'Sentiment Analysis accuracy: {dev_sa_acc:.3f}')
        print(f'Paraphrase Detection accuracy: {dev_pd_acc:.3f}')
        print(f'Semantic Textual Similarity correlation: {dev_sts_corr:.3f}\n')

        dev_acc = (dev_sa_acc + dev_pd_acc + (dev_sts_corr + 1) / 2) / 3

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('save the model..')
            # model.save_pretrained(args.lorapath) # Lora
            # save_config(args, config, args.configpath)
            patience = 0
            save_model(model, optimizer, args, config, args.filepath) # torch
        else:
            patience = patience + 1
        
        print(f'epoch {epoch} loss: {dev_acc:.3f}')
        print()
        print('='*200)
        print()

        if patience == 3:
            break
    
    return model



def test_multitask(args, model_0):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])

        # Init model.
        # config = {'hidden_dropout_prob': args.hidden_dropout_prob,
        #         'num_labels': num_labels,
        #         'hidden_size': 768,
        #         'data_dir': '.',
        #         'fine_tune_mode': args.fine_tune_mode}

        # config = SimpleNamespace(**config)

        # model = MultitaskBERT(config)
        # model = PeftModel.from_pretrained(model, args.lorapath, is_trainable=False)
        model = model.to(device)


        print(f"Loaded model to test from {args.filepath}")

        sa_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sa_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sa_test_data = SentenceClassificationTestDataset(sa_test_data, args)
        sa_dev_data = SentenceClassificationDataset(sa_dev_data, args)

        sa_test_dataloader = DataLoader(sa_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sa_test_data.collate_fn)
        sa_dev_dataloader = DataLoader(sa_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sa_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
        

        if args.debug:
            N = args.truncate_size
            para_dev_dataloader = truncate_dataloader(para_dev_dataloader, N)
            sts_dev_dataloader = truncate_dataloader(sts_dev_dataloader, N)
            sa_dev_dataloader = truncate_dataloader(sa_dev_dataloader, N)
        

        dev_sa_acc, dev_pd_acc, dev_sts_corr = model_eval_multitask(sa_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, 
                                                                    device, args)

        print(f'Sentiment Analysis accuracy: {dev_sa_acc:.3f}')
        print(f'Paraphrase Detection accuracy: {dev_pd_acc:.3f}')
        print(f'Semantic Textual Similarity correlation: {dev_sts_corr:.3f}\n')

        with open('/root/autodl-fs/dev_log.txt', "w+") as f:
            f.write(f"dev sentiment acc :: {dev_sa_acc :.3f}\n")
            f.write(f'dev paraphrase detection acc :: {dev_pd_acc:.3f}\n')
            f.write(f'dev sts corr :: {dev_sts_corr:.3f}\n')

        # dev_sa_acc, dev_pd_acc, dev_sts_corr = model_eval_multitask(sa_dev_dataloader,
        #                                                             para_dev_dataloader,
        #                                                             sts_dev_dataloader, model_0, device)

        # print('loss of model_0:')
        # print(f'Sentiment Analysis accuracy: {dev_sa_acc:.3f}')
        # print(f'Paraphrase Detection accuracy: {dev_pd_acc:.3f}')
        # print(f'Semantic Textual Similarity correlation: {dev_sts_corr:.3f}\n')


        # test_sst_y_pred, \
        #     test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
        #         model_eval_test_multitask(sa_test_dataloader,
        #                                   para_test_dataloader,
        #                                   sts_test_dataloader, model, device)

        # with open(args.sst_dev_out, "w+") as f:
        #     print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
        #     f.write(f"id \t Predicted_Sentiment \n")
        #     for p, s in zip(dev_sa_sent_ids, dev_sa_y_pred):
        #         f.write(f"{p} , {s} \n")

        # with open(args.sst_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Sentiment \n")
        #     for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
        #         f.write(f"{p} , {s} \n")

        # with open(args.para_dev_out, "w+") as f:
        #     print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
        #     f.write(f"id \t Predicted_Is_Paraphrase \n")
        #     for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
        #         f.write(f"{p} , {s} \n")

        # with open(args.para_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Is_Paraphrase \n")
        #     for p, s in zip(test_para_sent_ids, test_para_y_pred):
        #         f.write(f"{p} , {s} \n")

        # with open(args.sts_dev_out, "w+") as f:
        #     print(f"dev sts corr :: {dev_sts_corr :.3f}")
        #     f.write(f"id \t Predicted_Similiary \n")
        #     for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
        #         f.write(f"{p} , {s} \n")

        # with open(args.sts_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Similiary \n")
        #     for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
        #         f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--truncate_size", type=int, default=1)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--output_dir", type=str, help="img output diractory", default="./")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    args.lorapath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-lora' # Save config path
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    model = train_multitask(args)
    test_multitask(args, model)
    # os.system('shutdown')