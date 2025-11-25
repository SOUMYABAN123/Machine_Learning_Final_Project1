import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer




@dataclass
class DataIngesionConfig:
    train_data_path : str = os.path.join('artifact',"train.csv")
    test_data_path : str = os.path.join('artifact',"test.csv")
    raw_data_path : str = os.path.join('artifact',"data.csv")


class DataIngesion:
    def __init__(self):
        self.ingesion_config = DataIngesionConfig()
    
    def initiate_dataingesion(self):
        logging.info("Entered the data ingesion method or component")
        try:
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))
            logging.info('Exported or read the dataset as data frame')

            os.makedirs(os.path.dirname(self.ingesion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingesion_config.raw_data_path, index= False, header = True)

            logging.info('Train Test split initiated')
            train_set, test_set= train_test_split(df,test_size=0.2, random_state= 42)
            train_set.to_csv(self.ingesion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingesion_config.test_data_path, index= False, header = True)

            logging.info('Ingesion of the data is completed')

            return(
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path,
                
            )


        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    obj = DataIngesion()
    train_data, test_data = obj.initiate_dataingesion()

    data_transform = DataTransformation()
    train_arr,test_arr,_=data_transform.initiate_data_transform(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))



