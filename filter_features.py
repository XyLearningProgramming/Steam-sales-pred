import pandas as pd

class joined_tables():
    def __init__(self):
        self._joined_all = None
        self._read_all = False

    @property
    def join_all(self):
        if self._joined_all is None:
            print("Haven't set anything yet. Just return table steam")
            return self.steam
        return self._joined_all

    @join_all.setter
    def join_all(self,join_list=['tag']):
        '''
        join_list only accepts steam, desc, med,req,sup,tag

        :param str:
        :return:
        '''
        if self._read_all == False:
            self.read_all()
        table_dict = {'des':self.des,'med':self.med,
                      'req':self.req,'sup':self.req,
                      'tag':self.tag}
        for table_name in join_list:
            assert table_name[:3] in list(table_dict.keys())
        df = self.steam.copy()
        for table_name in join_list:
            df = pd.merge(df,table_dict[table_name],on='appid',how='left')
        self._joined_all = df

    def read_all(self):
        self._read_all = True
        self.steam = pd.read_csv('steam.csv')
        self.des = pd.read_csv('steam_description_data.csv')
        self.med = pd.read_csv('steam_media_data.csv')
        self.req = pd.read_csv('steam_requirements_data.csv')
        self.sup = pd.read_csv('steam_support_info.csv')
        self.tag = pd.read_csv('steamspy_tag_data.csv')

if __name__ =='__main__':
    tables = joined_tables()
    tables.join_all = ['tag']
    df = tables.join_all