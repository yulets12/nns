import pandas as pd

div = 12
mode = 'train'
df = pd.read_csv('../'+mode+'.csv')
print(df['Class Index'].unique())
symbols = ['.', ',', '/', '\\', '\\n', '"', ';', ':', ')', '(', '&', '!', '?', '-', '#', '@', "'", '<', '>']
classes = []
titles = []
for j in range(int(len(df['Title'])/div)):
    classes.append(df['Class Index'][j])
    titles.append(df['Title'][j])
    for symb in symbols:
        titles[j] = titles[j].replace(symb, ' ')
    for symb in [' '*x for x in range(2, 11)]:
        titles[j] = titles[j].replace(symb, ' ')
    titles[j] = titles[j].strip().lower()
descrs = []
for j in range(int(len(df['Description'])/div)):
    descrs.append(df['Description'][j])
    for symb in symbols:
        descrs[j] = descrs[j].replace(symb, ' ')
    for symb in [' '*x for x in range(2, 11)]:
        descrs[j] = descrs[j].replace(symb, ' ')
    descrs[j] = descrs[j].strip().lower()

new_df = pd.DataFrame({'Class Index': classes, 'Title': titles, 'Description': descrs})
#print(new_df)
new_df.to_csv('../prep_'+mode+'.csv', index=False)
