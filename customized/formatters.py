import pandas as pd


class OutputFormatter:
    def format_output(self, out):
        # convert to dataframe
        return pd.DataFrame(out)
