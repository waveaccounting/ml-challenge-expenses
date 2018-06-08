import pandas as pd

def Identify_Duplicate_Employees(Input_File_With_Path):
    employees = pd.DataFrame.from_csv(Input_File_With_Path,index_col=None,encoding='utf-8');
    employees['is_duplicated'] = employees.duplicated(['employee name','employee address','role'])
    g = employees.groupby(['employee name','employee address','role'])
    df1 = employees.set_index(['employee name','employee address','role'])
    employees['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])
    return employees

def Get_Clean_Dataframes(Employees_DF):
    Index_2_Replace = Employees_DF[Employees_DF['is_duplicated']==True][['employee id','dup_index']]
    return Index_2_Replace