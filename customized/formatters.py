import pandas as pd
import numpy as np


def _convert_output(out):
    # convert 2D output to 1D a single class label (4000 nodes into a single number per output)
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


class OutputFormatter:
    def format_output(self, out):

        labels = _convert_output(out)

        # add an index column
        df = pd.DataFrame(labels).reset_index(drop=False)
        # print(df.columns)
        # change column names
        df = df.rename(columns={0: "label", 'index': 'idprefix'})

        # add .jpg to the id column
        df = df.astype({'idprefix': 'str'})
        df['idsuffix'] = '.jpg'
        df['id'] = df['idprefix'] + df['idsuffix']

        # drop extra columns generated
        df = df.drop(['idprefix', 'idsuffix'], axis=1)

        # ensure id is first column
        df = df[['id','label']]

        return df
