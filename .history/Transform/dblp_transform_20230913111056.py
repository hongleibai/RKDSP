import argparse
from dblp_transform_model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, type=str, help='Targeting dataset.',
                        choices=['DBLP','DBLP2','PubMed','ACM','ACM2','Freebase'])
    parser.add_argument('-model', required=True, type=str, help='Targeting model.',
                        choices=['metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT',
                                 'TransE', 'DistMult', 'ComplEx', 'ConvE','RKDSP','HDGI','OpenNe','NSHE'])
    parser.add_argument('-attributed', required=True, type=str,
                        help='Only R-GCN, HAN, MAGNN, and HGT support attributed training.',
                        choices=['True', 'False'])
    parser.add_argument('-supervised', required=True, type=str,
                        help='Only R-GCN, HAN, MAGNN, and HGT support semi-supervised training.',
                        choices=['True', 'False'])
    parser.add_argument('-version',type=str)

    return parser.parse_args()


def check(args):
    if args.attributed == 'True':
        if args.model not in ['R-GCN', 'HAN', 'MAGNN', 'HGT','RKDSP','HDGI','NSHE']:
            print(f'{args.model} does not support attributed training!')
            print('Only R-GCN, HAN, MAGNN, and HGT support attributed training!')
            return False
        if args.dataset not in ['DBLP','DBLP2','PubMed','ACM','ACM2','Freebase']:
            print(f'{args.dataset} does not support attributed training!')
            print('Only DBLP and PubMed support attributed training!')
            return False

    if args.supervised == 'True':
        if args.model not in ['R-GCN', 'HAN', 'MAGNN', 'HGT']:
            print(f'{args.model} does not support semi-supervised training!')
            print('Only R-GCN, HAN, MAGNN, and HGT support semi-supervised training!')
            return False

    return True


def main():

    args = parse_args()

    if not check(args):
        return

    print('Transforming {} to {} input format for {}, {} training!'
          .format(args.dataset, args.model,
               'attributed' if args.attributed=='True' else 'unattributed',
               'semi-supervised' if args.supervised=='True' else 'unsupervised'))

    if args.model=='metapath2vec-ESim': metapath2vec_esim_convert(args.dataset)
    elif args.model=='PTE': pte_convert(args.dataset)
    elif args.model=='HIN2Vec': hin2vec_convert(args.dataset)
    elif args.model=='AspEm': aspem_convert(args.dataset)
    elif args.model=='HEER': heer_convert(args.dataset)
    elif args.model=='R-GCN': rgcn_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='HAN': han_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='MAGNN': magnn_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='HGT': hgt_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='TransE': transe_convert(args.dataset)
    elif args.model=='DistMult': distmult_convert(args.dataset)
    elif args.model=='ComplEx': complex_convert(args.dataset)
    elif args.model=='ConvE': conve_convert(args.dataset)
    elif args.model=='RKDSP':RKDSP_link_convert(args.dataset,args.attributed,args.version,False)
    elif args.model=='HDGI':hdgi_convert(args.dataset,args.attributed)
    elif args.model=='OpenNe':openne_convert(args.dataset)
    elif args.model=='NSHE':nshe_convert(args.dataset)
        
    print('Data transformation finished!')
    
    return


if __name__=='__main__':
    main()