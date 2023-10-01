# test_dummy_fracsum
#### Тестовое задание №2 компании Belka Digital

а). Дополнить sklearn.dummy.DummyRegressor способностью выполнять ещё один вид
фиктивной подгонки: новая стратегия “fracsum” возвращает сумму дробных частей в обучающем
наборе аналогично тому, как реализованы стратегии “mean” и “median”. Документирование
своей части кода с помощью докстрингов. Юнит-тесты на свою часть кода. Продемонстрировать
работу на примере двух обучающих наборов с наблюдениями целевых переменных (observed
targets):

а1) Y = [ 0.5 , 1.3, -0.8 ] ;
а2) Y = [ 5, 3, -8 ].

Значения объясняющих переменных разрешается выбрать произвольно.

б). **Способен ли такой регрессор делать множественную регрессию?**
    *Да, способен, так как он не учитывает объясняющие переменные,
    поэтому их может быть любое количество, это предусмотрено в DummyRegressor.*

в). **Нужно ли обернуть его в sklearn.multioutput.MultiOutputRegressor , чтобы сделать многомерную регрессию?**
    *Нет, не нужно. DummyRegressor предусматривает возможность нескольких целевых переменных,
    соответственно отнаследованный от него класс тоже.*
