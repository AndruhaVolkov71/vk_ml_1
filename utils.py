import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_df_info(df, thr=0.5):
    # сделали индексы - навзания колонок
    new_df = pd.DataFrame(index=df.columns)

    # добавляем колонку с типами данных
    new_df['type'] = df.dtypes.values

    # добавляем колонку с количеством уникальных значений (включая Nan) в колонке
    new_df['unique'] = df.nunique(dropna=False).values

    # добавляем колонку с долей Nan в колонке (округляем до 3 знаков после запятой), но если получается чистый ноль то пишем 0
    new_df['nan_proportion'] = df.isna().mean().values.round(3)
    new_df.loc[new_df['nan_proportion'] == 0, 'nan_proportion'] = 0

    # добавляем колонку с долей нулей в колонке
    new_df['zero_proportion'] = (df == 0).mean().values.round(3)

    # добавляем колонку с долей пустых строк в колонке
    new_df['empty_strings_proportion'] = (df == '').mean().values.round(3)

    # добавляем колонку с самым часто встречающимся значением в колонке (и количество таких значений)
    new_df['vc_max'] = df.mode(dropna=True).values[0]


    # добавляем колонку с долей самого частого значения в колонке
    new_df['vc_max_proportion'] = (df == new_df['vc_max']).mean().values.round(3)


    # теперь сделаем две колонки с различными примерами значений из колонки
    for colomn in df.columns:
        unique_values = df[colomn].dropna().unique()
        if len(unique_values) > 1:
            new_df.loc[colomn, 'example1'] = unique_values[0]
            new_df.loc[colomn, 'example2'] = unique_values[1]
        elif len(unique_values) == 1:
            new_df.loc[colomn, 'example1'] = unique_values[0]
            new_df.loc[colomn, 'example2'] = "нет второго примера"
        else:
            new_df.loc[colomn, 'example1'] = "нет примеров"
            new_df.loc[colomn, 'example2'] = "нет примеров"

    
    # добавляем trash_score колонки: max([суммарная доля нанов, нулей и пустых строк], [`vc_max_proportion` if `vc_max_proportion` > thr else 0])
    colomn = new_df['vc_max_proportion']
    colomn[colomn <= thr] = 0
    trash_score = pd.DataFrame([colomn, new_df['nan_proportion'] + new_df['zero_proportion'] + new_df['empty_strings_proportion'], colomn]).max()
    new_df['trash_score'] = trash_score



    return new_df
    
def plot_density(df, hue, cols=None, drop_zero=False, max_cat_thr=20):
    
    # создаем категориальные колонки, у которых уникальных значений не больше max_cat_thr
    categorical_cols = df.nunique()[df.nunique() <= max_cat_thr].index
    # исключаем hue из categorical_cols
    categorical_cols = list(set(categorical_cols) - set([hue]))

    # создаем числовые колонки используя метод is_numeric_dtype
    numerical_cols = df.select_dtypes(include='number').columns
    # исключаем hue и categorical_cols из numerical_cols
    # сначала сделаем объединение двух списков, а потом вычтем из numerical_cols
    intersection = set(numerical_cols).intersection(set(categorical_cols))
    numerical_cols = list(set(numerical_cols) - intersection)
    numerical_cols.remove(hue)

    for column in numerical_cols:
        fig, ax = plt.subplot_mosaic('''abc''', figsize=(10, 15))
        fig.suptitle(column + " vs " + hue)
        sns.histplot(df[df[column] != 0], x = column, hue = hue, bins = 20, multiple='stack', element='step', \
                        stat='count', alpha=0.8, ax=ax['a'], legend=True)
        sns.boxenplot(df[df[column] != 0], x=hue, y=column, showfliers=False, legend=True, ax=ax['b'], hue=hue)
        sns.stripplot(df[df[column] != 0].sample(200), x=hue, y=column, legend=True, ax=ax['b'])

        # считаем для каждого значения колонки hue долю нулей и нанов в колонке column, оборачиваем в датафрейм и строим sns.barplot
        df1 = df[column].isna().groupby(df[hue]).mean().rename('val').reset_index().assign(what='NaN')
        df2 = df[column].eq(0).groupby(df[hue]).mean().rename('val').reset_index().assign(what='0')
        df3 = pd.concat([df1, df2])
        # вместо чистого нуля используйте что-то отрицательное (напр. -0.1 * [значение самого высокого бина])
        df3['val'] = df3['val'].replace(0, -0.1 * df3['val'].max())
        sns.barplot(data=df3, x='what', y='val', hue=hue, ax=ax['c'], edgecolor='black')
        

        ax['c'].tick_params('x', rotation=90)
        ax['c'].grid(True, axis='y')
        ax['c'].axhline(0, color='black', ls='--')
        
        # убираем подписи для оси y у всех графиков, кроме левого
        ax['b'].set_ylabel('')
        ax['c'].set_ylabel('')

        fig.set_size_inches(20, 5)
        plt.show()
        print("################################################################################")
    
    
    # построим sns.countplot относительно каждой категориальной колонки
    for column in categorical_cols:
        fig, ax = plt.subplots(1, 1)
        

        # делаем более жирные шрифты
        sns.set(font_scale=1.5)

        df = df.replace('', '<empty>')
        df = df.fillna('<NaN>')

        sns.countplot(data=df, x=column, hue=hue, ax=ax, stat='count', alpha=0.8, legend=True, edgecolor='black')
        ax.tick_params('x', rotation=90)
        ax.grid(True, axis='y')
        fig.suptitle(column + " vs " + hue)
        fig.set_size_inches(15, 5)

        # добавляем горизонтальную решетку (яркие горизонтальные линии на графике всегда улучшают восприятие)
        ax.grid(True, axis='y')

        

        plt.show()
        print("################################################################################")
