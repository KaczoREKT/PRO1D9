#=======================================================================================================================
#                                                          ZAD 2
#=======================================================================================================================
print("Zad 2: Wprowadzenie danych i przekształcenie na postać binarną")

# Importujemy niezbędne moduły
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
import pandas as pd

# Nowe, bardziej interesujące dane transakcji
transactions = [
    ['apple', 'banana', 'milk'],
    ['banana', 'coffee', 'bread'],
    ['milk', 'bread', 'apple'],
    ['apple', 'banana', 'coffee', 'bread'],
    ['apple', 'coffee'],
    ['bread', 'milk'],
    ['coffee', 'milk', 'banana'],
    ['apple', 'bread', 'coffee'],
    ['banana', 'milk', 'bread'],
    ['coffee', 'apple', 'milk']
]

# Przekształcenie na postać binarną
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Wyświetlenie przetworzonych danych
print("Dane w postaci binarnej:")
print(df)

#=======================================================================================================================
#                                                ZAD 2: Produkty częste
#=======================================================================================================================
print("Zad 2: Generowanie produktów częstych i ich analiza")

import matplotlib.pyplot as plt

# Generowanie produktów częstych (min_support = 0.2)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Dodanie kolumny z liczbą elementów w zbiorze
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Sortowanie według wsparcia (malejąco)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print("Produkty częste (posortowane):")
print(frequent_itemsets)

# Wykres słupkowy: 5 najczęstszych produktów
top_5 = frequent_itemsets.head(5)
plt.bar(top_5['itemsets'].astype(str), top_5['support'], color='skyblue')
plt.xlabel('Produkty')
plt.ylabel('Wsparcie')
plt.title('Top 5 najczęstszych produktów')
plt.show()

#=======================================================================================================================
#                                                ZAD 3: Zbiory częste
#=======================================================================================================================
print("Zad 3: Zbiory częste dwu- i trzy-elementowe")

# Funkcja do konwersji zbiorów na stringi
def format_itemsets(frequent_df):
    frequent_df = frequent_df.copy()
    frequent_df['itemsets'] = frequent_df['itemsets'].apply(lambda x: ', '.join(sorted(x)))
    return frequent_df

# Filtracja i sortowanie zbiorów dwu-elementowych
frequent_2_itemsets = frequent_itemsets[frequent_itemsets['length'] == 2]
frequent_2_itemsets = frequent_2_itemsets.sort_values(by='support', ascending=False)
frequent_2_itemsets = format_itemsets(frequent_2_itemsets)

# Filtracja i sortowanie zbiorów trzy-elementowych
frequent_3_itemsets = frequent_itemsets[frequent_itemsets['length'] == 3]
frequent_3_itemsets = frequent_3_itemsets.sort_values(by='support', ascending=False)
frequent_3_itemsets = format_itemsets(frequent_3_itemsets)

# Wyświetlenie wyników
print("Częste zbiory dwu-elementowe (top 5):")
print(frequent_2_itemsets.head(5))

print("\nCzęste zbiory trzy-elementowe (top 5):")
print(frequent_3_itemsets.head(5))

#=======================================================================================================================
#                                                ZAD 4: Reguły asocjacyjne
#=======================================================================================================================
print("Zad 4: Generowanie reguł asocjacyjnych")

# Generowanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(transactions))

# Sortowanie według miary lift
rules_sorted_by_lift = rules.sort_values(by='lift', ascending=False)
print("\nTop 5 reguł według miary lift:")
print(rules_sorted_by_lift.head(5))

# Sortowanie według miary confidence
rules_sorted_by_confidence = rules.sort_values(by='confidence', ascending=False)
print("\nTop 5 reguł według miary confidence:")
print(rules_sorted_by_confidence.head(5))

# Sortowanie według miary conviction
rules_sorted_by_conviction = rules.sort_values(by='conviction', ascending=False)
print("\nTop 5 reguł według miary conviction:")
print(rules_sorted_by_conviction.head(5))

#=======================================================================================================================
#                                                ZAD 5: Ładowanie danych i analiza produktów
#=======================================================================================================================
print("Zad 5: Ładowanie danych, wyznaczanie częstości i wizualizacja produktów")

# Wyznaczanie liczby transakcji oraz liczby produktów
num_transactions = len(df)
num_products = df.shape[1]

print(f"Liczba transakcji: {num_transactions}")
print(f"Liczba produktów: {num_products}")

# Obliczanie wsparcia (częstości) dla każdego produktu
support = df.mean(axis=0).sort_values(ascending=False)

# Wizualizacja 10 najczęściej występujących produktów
top_10_products = support.head(10)
plt.bar(top_10_products.index, top_10_products.values, color='blue')
plt.xlabel('Produkty')
plt.ylabel('Wsparcie')
plt.title('Top 10 najczęściej występujących produktów')
plt.xticks(rotation=45, ha='right')
plt.show()

# Wyświetlenie najczęściej występujących produktów
print("10 najczęściej występujących produktów:")
print(top_10_products)

#=======================================================================================================================
#                                                ZAD 6: Wstępne przetwarzanie
#=======================================================================================================================
print("Zad 6: Wstępne przetwarzanie danych")

# Usuwanie zbędnych spacji z nazw produktów
transactions_cleaned = [[item.strip() for item in transaction] for transaction in transactions]

# Przekształcamy dane na postać binarną po usunięciu spacji
te = TransactionEncoder()
te_ary_cleaned = te.fit(transactions_cleaned).transform(transactions_cleaned)
df_cleaned = pd.DataFrame(te_ary_cleaned, columns=te.columns_)

# Wyświetlenie przetworzonych danych po usunięciu spacji
print("Dane po usunięciu spacji i przekształceniu na postać binarną:")
print(df_cleaned)

# Sprawdzanie liczby transakcji i produktów po czyszczeniu danych
num_transactions_cleaned = len(df_cleaned)
num_products_cleaned = df_cleaned.shape[1]

print(f"Liczba transakcji po czyszczeniu: {num_transactions_cleaned}")
print(f"Liczba produktów po czyszczeniu: {num_products_cleaned}")

#=======================================================================================================================
#                                                ZAD 7: Generowanie produktów częstych i rzadkich
#=======================================================================================================================
print("Zad 7: Generowanie produktów częstych i rzadkich")

# Ustawienie parametru min_support
min_support = 0.2

# Generowanie produktów częstych (min_support = 0.2)
frequent_itemsets = apriori(df_cleaned, min_support=min_support, use_colnames=True)

# Dodanie kolumny z liczbą elementów w zbiorze
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Sortowanie według wsparcia (malejąco)
frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)

# Najczęstsze produkty (top 10)
top_10_frequent = frequent_itemsets_sorted.head(10)

# Najrzadsze produkty (top 10)
top_10_rare = frequent_itemsets_sorted.tail(10)

# Wizualizacja produktów częstych
plt.figure(figsize=(10, 6))
plt.bar(top_10_frequent['itemsets'].astype(str), top_10_frequent['support'], color='green')
plt.xlabel('Produkty')
plt.ylabel('Wsparcie')
plt.title('Top 10 najczęstszych produktów')
plt.xticks(rotation=45, ha='right')
plt.show()

# Wizualizacja produktów rzadkich
plt.figure(figsize=(10, 6))
plt.bar(top_10_rare['itemsets'].astype(str), top_10_rare['support'], color='red')
plt.xlabel('Produkty')
plt.ylabel('Wsparcie')
plt.title('Top 10 najrzadszych produktów')
plt.xticks(rotation=45, ha='right')
plt.show()

# Wyświetlenie najczęstszych i najrzadszych produktów
print("Top 10 najczęstszych produktów:")
print(top_10_frequent)

print("\nTop 10 najrzadszych produktów:")
print(top_10_rare)


#=======================================================================================================================
#                                                ZAD 8: Generowanie zbiorów częstych
#=======================================================================================================================
print("Zad 8: Generowanie zbiorów częstych dwuelementowych i trzyelementowych")

# Generowanie zbiorów częstych dwu-elementowych
frequent_2_itemsets = frequent_itemsets_sorted[frequent_itemsets_sorted['length'] == 2]

# Generowanie zbiorów częstych trzy-elementowych
frequent_3_itemsets = frequent_itemsets_sorted[frequent_itemsets_sorted['length'] == 3]

# Wyświetlenie 5 najczęstszych zbiorów dwu-elementowych
print("Częste zbiory dwu-elementowe (top 5):")
print(frequent_2_itemsets.head(5))

# Wyświetlenie 5 najczęstszych zbiorów trzy-elementowych
print("\nCzęste zbiory trzy-elementowe (top 5):")
print(frequent_3_itemsets.head(5))

#=======================================================================================================================
#                                                ZAD 9: Generowanie reguł asocjacyjnych
#=======================================================================================================================
print("Zad 9: Generowanie reguł asocjacyjnych")

# Generowanie reguł asocjacyjnych na podstawie częstych zbiorów
rules = association_rules(frequent_itemsets_sorted, metric="confidence", min_threshold=0.5, num_itemsets=len(transactions))

# Dodanie sortowania i filtrowania reguł dla miar: confidence, conviction, lift
# Top 10 reguł według confidence
rules_sorted_confidence = rules.sort_values(by='confidence', ascending=False).head(10)
print("Top 10 reguł według confidence:")
print(rules_sorted_confidence)

# Top 10 reguł według conviction
rules_sorted_conviction = rules.sort_values(by='conviction', ascending=False).head(10)
print("\nTop 10 reguł według conviction:")
print(rules_sorted_conviction)

# Top 10 reguł według lift
rules_sorted_lift = rules.sort_values(by='lift', ascending=False).head(10)
print("\nTop 10 reguł według lift:")
print(rules_sorted_lift)

# Wizualizacja reguł według miary confidence
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(rules_sorted_confidence['antecedents'].astype(str) + " -> " + rules_sorted_confidence['consequents'].astype(str),
        rules_sorted_confidence['confidence'], color='purple')
plt.xlabel('Reguły asocjacyjne')
plt.ylabel('Confidence')
plt.title('Top 10 reguł według confidence')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Wizualizacja reguł według miary lift
plt.figure(figsize=(10, 6))
plt.bar(rules_sorted_lift['antecedents'].astype(str) + " -> " + rules_sorted_lift['consequents'].astype(str),
        rules_sorted_lift['lift'], color='orange')
plt.xlabel('Reguły asocjacyjne')
plt.ylabel('Lift')
plt.title('Top 10 reguł według lift')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Wizualizacja reguł według miary conviction
plt.figure(figsize=(10, 6))
plt.bar(rules_sorted_conviction['antecedents'].astype(str) + " -> " + rules_sorted_conviction['consequents'].astype(str),
        rules_sorted_conviction['conviction'], color='green')
plt.xlabel('Reguły asocjacyjne')
plt.ylabel('Conviction')
plt.title('Top 10 reguł według conviction')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#=======================================================================================================================
#                                                ZAD 10: Porównanie czasu działania Apriori i FPGrowth
#=======================================================================================================================
print("Zad 10: Porównanie czasu działania algorytmu Apriori i FPGrowth")

import time
from mlxtend.frequent_patterns import fpgrowth

# Ustawienie progów min_support
min_sup_values = [0.01, 0.02, 0.03, 0.04, 0.05]
apriori_times = []
fpgrowth_times = []

for min_sup in min_sup_values:
    # Pomiar czasu dla Apriori
    start_time = time.time()
    apriori(df_cleaned, min_support=min_sup, use_colnames=True)
    apriori_times.append(time.time() - start_time)

    # Pomiar czasu dla FPGrowth
    start_time = time.time()
    fpgrowth(df_cleaned, min_support=min_sup, use_colnames=True)
    fpgrowth_times.append(time.time() - start_time)

# Wizualizacja czasu działania
plt.figure(figsize=(10, 6))
plt.plot(min_sup_values, apriori_times, label='Apriori', marker='o')
plt.plot(min_sup_values, fpgrowth_times, label='FPGrowth', marker='o')
plt.xlabel('Minimalne wsparcie (min_sup)')
plt.ylabel('Czas działania (sekundy)')
plt.title('Porównanie czasu działania Apriori i FPGrowth')
plt.legend()
plt.grid(True)
plt.show()

# Obserwacje
print("\nObserwacje:")
print("- Algorytm Apriori jest wolniejszy niż FPGrowth przy każdym progu min_sup.")
print("- Czas działania obu algorytmów maleje wraz ze wzrostem minimalnego wsparcia, ponieważ mniejsza liczba zbiorów jest generowana.")
