import torch
import argparse, os

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv6_consistency import train_mv6_consistency

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth, cosine_sim_for_simclr

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def naming_convention(args):
    if args.eq_ge:
        name = "bc_eqge_split={}.pt"
    elif args.eq_pret:
        name = "bc_eqpret_split={}.pt"
    elif args.ge_rand_init:
        name = "bc_gerand_split={}.pt"
    elif args.no_ste:
        name = "bc_noste_split={}.pt"
    elif args.simclr:
        name = "bc_simclr_split={}.pt"
    elif args.no_la:
        name = "bc_nola_split={}.pt"
    elif args.no_con:
        name = "bc_nocon_split={}.pt"
    elif args.cnn:
        name = "bc_cnn_split={}.pt"
    elif args.lstm:
        name = "bc_lstm_split={}.pt"
    elif args.rvalue is not None:
        name = "bc_r={:.2f}".format(args.rvalue) + "_split={}.pt"
    else:
        name = 'bc_full_split={}.pt'
    
    if args.lam != 1.0:
        # Not included in ablation parameters or other, so name it;
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

def main(args):

    #tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/formal_models/Scomb_transformer_split={}.pt"
    tencoder_path =args.tencoder_path

    if args.lstm:
        arch = 'lstm'
        tencoder_path = "/content/Ash_TimeX/models/Scomb_transformer_split={}_cpu.pt"
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/models/Scomb_cnn_split={}.pt"
    else:
        arch = 'transformer'
        #tencoder_path = "/content/Ash_TimeX/models/Scomb_transformer_split={}_cpu.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 4,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training':True, 'num_negatives_simclr':64}
    else:
        sim_criterion_cons = EmbedConsistencyLoss()
        sc_expand_args = {'simclr_training':False, 'num_negatives_simclr':64}

    if args.no_la:
        sim_criterion = sim_criterion_cons
    elif args.no_con:
        sim_criterion = sim_criterion_label
    else: # Regular
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        if args.simclr:
            selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam = 1.0)
        else:
            selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)
        #selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)

    targs = transformer_default_args

    for i in range(4, 6):
        D = process_Synth(split_no = i, device = device, base_path = '/content/Ash_TimeX/data/singleuv')
        dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
        train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

        val, test = D['val'], D['test']

        # Calc statistics for baseline:
        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased = True, dim = 1)

        # Change transformer args:
        targs['trans_dim_feedforward'] = 64
        targs['trans_dropout'] = 0.25
        targs['nlayers'] = 2
        targs['norm_embedding'] = False

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
            archtype = arch,
            cnn_dim = 128
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        model = TimeXModel(
            d_inp = 1,
            max_len = 200,
            n_classes = 4,
            n_prototypes = 50,
            gsat_r = 0.5 if (args.rvalue is None) else args.rvalue,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            masktoken_stats = (mu, std),
        )

        model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        model.to(device)

        model.init_prototypes(train = (D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device)))

        if not args.ge_rand_init: # Copies if not running this ablation
            model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
        
        model_suffix = naming_convention(args)
        spath = os.path.join('models', model_suffix)
        spath = spath.format(i)
        print('saving at', spath)

        best_model = train_mv6_consistency(
            model,
            optimizer = optimizer,
            train_loader = train_loader,
            clf_criterion = clf_criterion,
            sim_criterion = sim_criterion,
            beta_exp = 2.0,
            beta_sim = 1.0,
            val_tuple = val, 
            num_epochs = 50,
            save_path = spath,
            train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
            early_stopping = True,
            selection_criterion = selection_criterion,
            label_matching = True,
            embedding_matching = True,
            use_scheduler = True,
            **sc_expand_args
        )

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        f1, _ = eval_mv4(test, model)
        print('Test F1: {:.4f}'.format(f1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action = 'store_true', help = 'G = G_E')
    ablations.add_argument('--eq_pret', action = 'store_true', help = 'G_pret = G')
    ablations.add_argument('--ge_rand_init', action = 'store_true', help = "Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action = 'store_true', help = 'Does not use STE')
    ablations.add_argument('--simclr', action = 'store_true', help = 'Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action = 'store_true', help = 'No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action = 'store_true', help = 'No consistency loss - just label')
    # Note if you don't activate any of them, it just trains the normal method

    ablations.add_argument('--lstm', action = 'store_true', help = 'Run w LSTM')
    ablations.add_argument('--cnn', action = 'store_true')

    parser.add_argument('--rvalue', type = float, default = None, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')

    parser.add_argument('--tencoder_path' ,default="/content/Ash_TimeX/models/Scomb_transformer_split={}_cpu.pt")

    args = parser.parse_args()

    main(args)
