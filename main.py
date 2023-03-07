from argparse import ArgumentParser
import logging
from typing import Optional

from trainers.decission_tree import DecissionTreeTrainer
from trainers.elastic_net import ElasticNetTrainer

logger = logging.getLogger(__name__)


def main(experiment_name: Optional[str],
         data_path: Optional[str],
         model_name: Optional[str],
         registered_model_name: Optional[str],
         registered_model_version_stage: Optional[str],
         save_signature: Optional[bool],
         max_depth: Optional[int],
         max_leaf_nodes: Optional[int],
         alpha: Optional[int],
         l1_ratio: Optional[int],
         plot_file_dt: Optional[str],
         plot_file_en: Optional[str],
         output_path: Optional[str]):
    """
    Train the model.
    """
    logger.info("Start trainning Experiment name")
    logger.debug("Options:")

    for k, v in locals().items(): 
        logger.debug("  %s: %s", k, v)
    
    dt_trainer = DecissionTreeTrainer(experiment_name=experiment_name,
                                   data_path=data_path,
                                   save_signature=save_signature,
                                   registered_model_version_stage=registered_model_version_stage,  # noqa
                                   output_path=output_path)
    
    
    dt_trainer.train(registered_model_name=registered_model_name,
                  plot_file=plot_file_dt,
                  max_depth=max_depth,
                  max_leaf_nodes=max_leaf_nodes)
    
    en_trainer = ElasticNetTrainer(experiment_name=experiment_name,
                                   data_path=data_path,
                                   save_signature=save_signature,
                                   registered_model_version_stage=registered_model_version_stage,  # noqa
                                   output_path=output_path)
    
    
    en_trainer.train(registered_model_name=registered_model_name,
                     plot_file=plot_file_en,
                     alpha=alpha,
                     l1_ratio=l1_ratio)


def build_parser() -> ArgumentParser:
    """
    Define arguments for this script.
    """
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="sklearn", required=False) # noqa
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="./wine-quality.csv", required=False) # noqa
    parser.add_argument("--registered_model_version_stage", dest="registered_model_version_stage", help="registered_model_version_stage", default="None", required=False) # noqa
    parser.add_argument("--save_signature", dest="save_signature", help="save_signature", type=bool, default=False, required=False) # noqa
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=10, type=int, required=False) # noqa
    parser.add_argument("--alpha", dest="alpha", help="alpha", default=0.5, type=int, required=False) # noqa
    parser.add_argument("--l1_ratio", dest="l1_ratio", help="l1_ratio", default=0.5, type=int, required=False) # noqa
    parser.add_argument("--max_leaf_nodes", dest="max_leaf_nodes", help="max_leaf_nodes", default=100, type=int, required=False) # noqa
    parser.add_argument("--output_path", dest="output_path", help="output_path", default=None, required=False) # noqa
    parser.add_argument("--plot_file_dt", dest="plot_file_dt", help="plot_file_dt", default="plots/plot_dt.png", required=False) # noqa
    parser.add_argument("--plot_file_en", dest="plot_file_en", help="plot_file_en", default="plots/plot_en.png", required=False) # noqa
    parser.add_argument("--model_name", dest="model_name", help="model_name", default="default_model_name", required=False) # noqa
    parser.add_argument("--registered_model_name", dest="registered_model_name", help="registered_model_name", default="DecisionTree model", required=False) # noqa
    return parser


if __name__ == "__main__":
    
    args = build_parser().parse_args()

    main(experiment_name=args.experiment_name,
         data_path=args.data_path,
         model_name=args.model_name,
         registered_model_name=args.registered_model_name,
         registered_model_version_stage=args.registered_model_version_stage,
         save_signature=args.save_signature,
         max_depth=args.max_depth,
         alpha=args.alpha,
         l1_ratio=args.l1_ratio,
         max_leaf_nodes=args.max_leaf_nodes,
         plot_file_dt=args.plot_file_dt,
         plot_file_en=args.plot_file_en,
         output_path=args.output_path)