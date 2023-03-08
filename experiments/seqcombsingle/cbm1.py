import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, L1Loss_permask, PairwiseDecorrelation
from txai.trainers.train_cbmv1 import train_cbmv1
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.cbmv1 import CBMv1
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_cbmv1
from txai.synth_data.simple_spike import SpikeTrainDataset

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

exp_criterion = [L1Loss_permask(norm=True), PairwiseDecorrelation()]


def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.eig(mat)[0][:,0]>=0).all())


for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    model = CBMv1(
        d_inp = 1,
        max_len = 50,
        n_classes = 4,
        n_concepts = 2,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        norm_embedding = False,
        #type_masktoken = 'norm_datawide',
        type_archmask = 'attn',
        masktoken_kwargs = {'mu': mu, 'std': std},
        mask_seed = 1234,
        smooth_concepts = False,
        distance_method = 'mahalanobis',
    )
    model.to(device)
    model.encoder.load_state_dict(torch.load('models/Scomb_transformer_split={}.pt'.format(i)))

    whole, times, batch_id = gen_dataset(template = mu.unsqueeze(1), samps=1000, device = device)
    cbank = (whole, times, batch_id)
    model.store_concept_bank(whole, times, batch_id)

    #model.freeze_encoder(freeze_mlp = True)
    #model.freeze_mlp_dist()

    #print(model.distance_mlp.get_parameter('0.weight'))
    #print(model.frozen_mlp)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.01)
    
    spath = 'models/Scomb_cbm_attn_split={}.pt'.format(i)

    best_model = train_cbmv1(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        exp_criterion = exp_criterion,
        beta = torch.tensor([0.25, 0.5]),
        val_tuple = val, 
        num_epochs = 300,
        save_path = spath,
        recompute_cbank_steps = 5,
        concept_bank = cbank,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)
    if model.distance_method == 'centroid':
        model.mu = torch.load('concept_bank_centroid.pt')['mu']

    if model.distance_method == 'mahalanobis':
        model.load_concept_bank('concept_bank_new.pt')

    f1, _ = eval_cbmv1(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()