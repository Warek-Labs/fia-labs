from shared import *

df = pd.read_csv('labels.csv')
df = df.sample(frac=1).reset_index(drop=True)

train_df_size = int(len(df) * 0.65)
train_df: pd.DataFrame = df[0:train_df_size]

validation_df_size = int(len(df) * 0.20)
validation_df_start = train_df_size
validation_df_end = validation_df_start + validation_df_size
validation_df: pd.DataFrame = df[train_df_size:validation_df_end]

test_df_size = int(len(df) * 0.20)
test_df_start = validation_df_end
test_df_end = test_df_start + test_df_size
test_df: pd.DataFrame = df[test_df_start:test_df_end]

DATASETS_PATH = 'datasets'
TRAIN_DATA_PATH = 'datasets/train'
TEST_DATA_PATH = 'datasets/test'
VALIDATION_DATA_PATH = 'datasets/validation'

if not os.path.exists(DATASETS_PATH):
   mode = 0o777
   os.mkdir(DATASETS_PATH, mode)

   for path in [TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH]:
      os.mkdir(path, mode)
      os.mkdir(f'{path}/good', mode)
      os.mkdir(f'{path}/bad', mode)

   for df, dataset_path in [(train_df, TRAIN_DATA_PATH), (test_df, TEST_DATA_PATH),
                            (validation_df, VALIDATION_DATA_PATH)]:
      for index, row in df.iterrows():
         path = row['new_path']
         label = row['label']
         shutil.copy(path, f'{dataset_path}/{"good" if label is True else "bad"}')

else:
   print(f'{DATASETS_PATH} folder already exists')
