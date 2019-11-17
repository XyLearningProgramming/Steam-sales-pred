import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

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

def k_means_playtime(df):
    plt.scatter(df.average_playtime, df.median_playtime)
    plt.show()
    X = df.loc[:, ['average_playtime', 'median_playtime']].values
    # https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('Elbow_method.png')
    plt.show()

    # number of cluster is 3
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.savefig('kmeans_with_center.png')
    plt.show()
    return pred_y,kmeans.cluster_centers_


def analyze_residuals(df):
    # https://songhuiming.github.io/pages/2016/11/27/linear-regression-in-python-outliers-leverage-detect/
    # https://stackoverflow.com/questions/10231206/can-scipy-stats-identify-and-mask-obvious-outliers
    # Need to know more about Bonferroni-adjusted p-value

    # ols and identify outliers
    y = df['median_playtime']
    X = df['average_playtime']
    result = sm.OLS(y, sm.add_constant(X)).fit()
    print(f'The r-squared value is {str(result.rsquared)}')
    fig, ax = plt.subplots()
    sns.regplot(x=X, y=y, ax=ax)
    plt.savefig('Regression_plot.png')
    plt.show()

    fig, ax = plt.subplots()
    fig = sm.graphics.influence_plot(result, ax=ax, criterion='cooks')
    plt.savefig('cooks_dect_of_outliers.png')
    plt.show()

    # outliers
    # Test observations for outliers according to method
    # bonferroni : one-step correction
    # The unadjusted p-value is stats.t.sf(abs(resid), df)
    # where df = df_resid - 1.
    # stats.t.sf is A Student’s T continuous random variable's survival function, which is 1-cdf
    # actually we're using student's T's distribution as that of residuals and test outlier

    test = result.outlier_test()
    table_outliers = df.loc[:, ['appid', 'name', 'median_playtime', 'average_playtime']].copy()
    table_outliers = pd.concat([table_outliers, test], axis=1)

    # assign values to outliers
    table_outliers['cat'] = 0
    # left skewed median > mean
    table_outliers.cat = np.where((table_outliers.unadj_p < 0.5) & (table_outliers.student_resid > 0), 1,
                                  table_outliers.cat)
    # right skewed median < mean
    table_outliers.cat = np.where((table_outliers.unadj_p < 0.5) & (table_outliers.student_resid < 0), 2,
                                  table_outliers.cat)
    return table_outliers

def cat_game_on_times(df):
    pred_y,k_means_centers = k_means_playtime(df)
    k_means_centers = pd.DataFrame(k_means_centers)
    table_outliers = analyze_residuals(df)
    table_outliers['time_quant'] = pred_y
    writer = pd.ExcelWriter('game_categories.xlsx',engine='xlsxwriter')
    table_outliers.to_excel(writer,sheet_name='Sheet1')
    k_means_centers.to_excel(writer,index=False,sheet_name='Sheet2')
    writer.save()

if __name__ =='__main__':
    tables = joined_tables()
    tables.join_all = ['tag']
    df = tables.steam
    cat_game_on_times(df)



