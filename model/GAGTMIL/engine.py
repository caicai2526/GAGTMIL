#models/GAGTMIL/engine.py
import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored

import torch 
import torch.optim
from torch.nn.utils import clip_grad_norm_  
from torch.optim.lr_scheduler import ReduceLROnPlateau  


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        self.writer = None
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            os.makedirs(writer_dir, exist_ok=True)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_scores = -float("inf")
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if getattr(self.args, "resume", ""):  
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint.get('best_score', self.best_scores)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint.get('best_score', None)))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))


        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return
   
        early_stop = getattr(self.args, "early_stop", True)
        patience = int(getattr(self.args, "patience", 5))
        save_best = getattr(self.args, "save_best", True)
        bad_epochs = 0

        fold_dir = os.path.join(self.results_dir, 'fold_' + str(self.fold))
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch

            self.train(train_loader, model, criterion, optimizer)
            val_cindex, val_loss = self.validate(val_loader, model, criterion)
            improved = val_cindex > (self.best_scores + 1e-4) 
            if improved:
                self.best_scores = val_cindex
                self.best_epoch = self.epoch
                if save_best:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_score': self.best_scores
                    })
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(' *** best score={:.4f} at epoch {}'.format(self.best_scores, self.best_epoch))

   
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    monitor = getattr(self.args, "scheduler_monitor", "cindex")
                    if monitor == "loss":
                        scheduler.step(val_loss)  
                    else:
                        scheduler.step(val_cindex) 
                else:
                    scheduler.step()

            print('>>>')
            print('>>>')

            if early_stop and bad_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improv. for {patience} epochs).")
                break

        return self.best_scores, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()

        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch {}'.format(self.epoch))

        accum_steps = max(1, int(getattr(self.args, "accum_steps", 8)))
        clip_norm = float(getattr(self.args, "clip_grad_norm", 1.0))
        optimizer.zero_grad()

        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()

            drop_p = float(getattr(self.args, "bag_drop_p", 0.0))
            min_keep = int(getattr(self.args, "bag_min_keep", 64))
            if drop_p > 0.0:
                N = data_WSI.size(0)
                keep = min(N, max(1, max(min_keep, int(N * (1.0 - drop_p)))))
                if keep < N:
                    perm = torch.randperm(N, device=data_WSI.device)[:keep]
                    data_WSI = data_WSI[perm]
        

            # prediction
            hazards, S = model(data_WSI)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            rm = getattr(self.args, "risk_metric", "sum_hazard")
            if rm == "sum_hazard":
                risk = torch.sum(hazards, dim=1).detach().cpu().numpy()
            elif rm == "neg_logS":
                risk = (-torch.log(S.clamp(min=1e-7)).sum(dim=1)).detach().cpu().numpy()
            else:  # "sumS_neg"
                risk = (-torch.sum(S, dim=1)).detach().cpu().numpy()
            all_risk_scores[batch_idx] = float(risk)
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = float(data_Event)

            total_loss += loss.item()

            # backward with accumulation
            (loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if clip_norm > 0:
                    clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad()

        if (len(dataloader) > 0) and ((batch_idx + 1) % accum_steps != 0):
            if clip_norm > 0:
                clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        # calculate loss and error for each epoch
        loss = total_loss / max(1, len(dataloader))
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
        )[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)
             try:
                self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], self.epoch)
            except Exception:
                pass

    def validate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch {}'.format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            with torch.no_grad():
                hazards, S = model(data_WSI)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            total_loss += loss.item()
        
            rm = getattr(self.args, "risk_metric", "sum_hazard")
            if rm == "sum_hazard":
                risk = torch.sum(hazards, dim=1).detach().cpu().numpy()
            elif rm == "neg_logS":
                risk = (-torch.log(S.clamp(min=1e-7)).sum(dim=1)).detach().cpu().numpy()
            else:  # "sumS_neg"
                risk = (-torch.sum(S, dim=1)).detach().cpu().numpy()
            all_risk_scores[batch_idx] = float(risk)
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = float(data_Event)
  
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
        )[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', loss, self.epoch)
            self.writer.add_scalar('val/c_index', c_index, self.epoch)
        return c_index, loss

    def save_checkpoint(self, state):
 
            fold_dir = os.path.join(self.results_dir, 'fold_' + str(self.fold))
            os.makedirs(fold_dir, exist_ok=True)

            if self.filename_best is not None and os.path.isfile(self.filename_best):
                try:
                    os.remove(self.filename_best)
                except Exception:
                    pass
            self.filename_best = os.path.join(
                fold_dir,
                'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch'])
            )
            print('save best model {filename}'.format(filename=self.filename_best))
            torch.save(state, self.filename_best)
