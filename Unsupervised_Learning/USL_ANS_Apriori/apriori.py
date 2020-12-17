import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [['A', 'C', 'D'],
		   ['B', 'C', 'E'],
		   ['A', 'B', 'C', 'E'],
		   ['B', 'E'],
		   ['A', 'C', 'E']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary,columns=te.columns_)

items_frecuentes = apriori(df, min_support=0.4, use_colnames=True)
reglas = association_rules(items_frecuentes, metric="confidence", min_threshold=0.6)
reglas = reglas.sort_values(['confidence'], ascending=[False])
print(reglas.head())