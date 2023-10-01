import numpy as np
from sklearn.dummy import DummyRegressor


class DummyRegressorFracSum(DummyRegressor):
    """
    DummyRegressorFracSum расширяет модель sklearn.dummy.DummyRegressor
    стратегией “fracsum”,
    которая возвращает сумму дробных частей в обучающем наборе

    Параметры
    ----------
    Те же, что и у родительского класса DummyRegressor,
    плюс вариант стартегии "fracsum", установленный по умолчанию.
    strategy : {"mean", "median", "constant", "fracsum"}, default="fracsum"
    """

    def __init__(self, strategy="fracsum", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant,
                         quantile=quantile)

    def fit(self, X, y, sample_weight=None):
        """
        Обучение модели DummyRegressorFracSum.
        Для проверки входных данных сначала вызывается
        стратегия mean, затем результат заменяется нужным значением.

        """

        # Если стратегия "fracsum", то заменяем значение "constant_"
        # вычисленной суммой дробных частей, иначе
        # возвращаем метод fit родительского класса

        if self.strategy == "fracsum":
            self.strategy = "mean"
            super().fit(X, y)
            self.strategy = "fracsum"
            self.constant_ = np.sum(np.mod(y, 1), axis=0)
        else:
            super().fit(X, y, sample_weight=sample_weight)
        return self


if __name__ == '__main__':
    # Объясняющие переменные выбраны произвольным образом,
    # модель их игнорирует
    X = np.array([1, 2, 3])
    X_test = np.array([4, 5, 6, 7])

    # Вариант А1
    print('---A1---')
    # Целевые переменные
    Y = np.array([0.5, 1.3, -0.8])
    dummy_regressor = DummyRegressorFracSum()
    dummy_regressor.fit(X, Y)
    # Прогноз с другими объясняющими переменными
    Y_pred = dummy_regressor.predict(X_test)
    print('Y_pred =', Y_pred)

    # Вариант А2
    print('---A2---')
    # Целевые переменные
    Y = np.array([5, 3, -8])
    dummy_regressor = DummyRegressorFracSum()
    dummy_regressor.fit(X, Y)
    # Прогноз с другими объясняющими переменными
    Y_pred = dummy_regressor.predict(X_test)
    print('Y_pred =', Y_pred)
