class PostProcess:
    def __init__(self, name):
        self.name = name

    def match_rows(self, df, cols=None):
        duplicates_map = df.duplicated(subset=cols, keep=False)
        if cols:
            duplicates_map = duplicates_map[cols]
        groups = duplicates_map.groupby(list(duplicates_map)).apply(lambda x: tuple(x.index)).tolist()
        return groups


class AverageCyclesPostprocess(PostProcess):
    def __init__(self):
        super().__init__("average_cycles")

    def process(self, df):
        groups = self.match_rows(df, cols=["Model", "Backend", "Target", "Features", "Config"])
        df_new = pd.DataFrame(columns=df.columns)
        for group in groups:
            group = list(group)
            sub_df = df.iloc[group]
            first = sub_df.iloc[0]
            nums = sub_df["Num"]
            cycles = sub_df["Cycles"]
            avg_cycles = cycles / nums
            first["Cycles"] = avg_cycles
            df_new = pd.concat([df_new, first])
        df_new.drop(columns=["Num"])
        return df


class DetailedCyclesPostprocess(PostProcess):
    def __init__(self):
        super().__init__("detailed_cycles")

    def process(self, df):
        return NotImplementedError
