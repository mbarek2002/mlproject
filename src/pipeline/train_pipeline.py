import sys

from src.exception import CustomException
from src.logger import logging
from src.pipeline.stages import ingest, train, transform


class TrainPipeline:
    '''
    Runs the 3 stages in a single process, without DVC.
    With DVC, prefer "dvc repro": it re-runs only the stages whose inputs changed.
    '''

    def run(self):
        try:
            logging.info("Training pipeline started")
            ingest()
            transform()
            return train()

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(TrainPipeline().run())
