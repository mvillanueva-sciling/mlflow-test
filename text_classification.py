from argparse import ArgumentParser
import logging
from typing import Optional

from trainers.svm_classifier import SvmClassifier
logger = logging.getLogger(__name__)


def main(experiment_name: Optional[str],
         train_path: Optional[str],
         model_name: Optional[str],
         registered_model_name: Optional[str],
         registered_model_version_stage: Optional[str],
         save_signature: Optional[bool],
         plot_file: Optional[str],
         output_path: Optional[str]):
    """
    Train the model.
    """
    logger.info("Start trainning Experiment name")
    logger.debug("Options:")

    for k, v in locals().items(): 
        logger.debug("  %s: %s", k, v)
    
    dt_trainer = SvmClassifier(experiment_name=experiment_name,
                                   train_path=train_path,
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
    parser.add_argument("--train_path", dest="train_path", help="train_path", default="data/corpus.csv", required=False) # noqa
    parser.add_argument("--registered_model_version_stage", dest="registered_model_version_stage", help="registered_model_version_stage", default="None", required=False) # noqa
    parser.add_argument("--save_signature", dest="save_signature", help="save_signature", type=bool, default=False, required=False) # noqa
    parser.add_argument("--output_path", dest="output_path", help="output_path", default=None, required=False) # noqa
    parser.add_argument("--plot_file", dest="plot_file", help="plot_file", default="plots/plot.png", required=False) # noqa
    parser.add_argument("--model_name", dest="model_name", help="model_name", default="default_model_name", required=False) # noqa
    parser.add_argument("--registered_model_name", dest="registered_model_name", help="registered_model_name", default="DecisionTree model", required=False) # noqa
    return parser


if __name__ == "__main__":
    
    args = build_parser().parse_args()

    main(experiment_name=args.experiment_name,
         train_path=args.train_path,
         model_name=args.model_name,
         registered_model_name=args.registered_model_name,
         registered_model_version_stage=args.registered_model_version_stage,
         save_signature=args.save_signature,
         plot_file=args.plot_file,
         output_path=args.output_path)