from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from schemas.adapters.mappers import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    # (a) Transformacion Numericas: imputa nulos con la mediana
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # ("scaler", StandardScaler()),  # opcional futuro
    ])
    # (b) Transformacion Categoria: imputa nulos + OneHotEncoder
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # opcional: nombres más limpios
    )

    return preprocessor
