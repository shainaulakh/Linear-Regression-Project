from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split features and target into training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

def scale_data(X_train, X_test):
    """
    Scale training and testing features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler