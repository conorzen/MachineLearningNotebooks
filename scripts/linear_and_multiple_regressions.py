import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("data_csv/tennis_stats.csv")

print(df.head())
print(df.info())


# create scatter plots here:
def plot_scatter(x, y, title, x_label, y_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


# view scatter plots here:
plot_scatter(
    df["BreakPointsOpportunities"],
    df["Winnings"],
    "Break Points Opportunities vs Winnings",
    "Break Points Opportunities",
    "Winnings",
)
plot_scatter(
    df["ReturnPointsWon"],
    df["Winnings"],
    "Return Points Won vs Winnings",
    "Return Points Won",
    "Winnings",
)
plot_scatter(
    df["FirstServeReturnPointsWon"],
    df["Winnings"],
    "First Serve Return Points Won vs Winnings",
    "First Serve Return Points Won",
    "Winnings",
)
plot_scatter(
    df["TotalPointsWon"],
    df["Winnings"],
    "Total Points Won vs Winnings",
    "Total Points Won",
    "Winnings",
)

# inital observations: BreakPointsOpportunities seem to have a positive correlation with Winnings.
# returnPointsWon seems to have a slightly positive correlation with Winnings but not as strong as BreakPointsOpportunities.

## perform single feature linear regressions here:


def regression_model(features, output, title, x_label, y_label):
    # split the data into training and testing sets here:
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2)

    # create and fit the model here:
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f"The coefficient: {model.coef_[0]}")
    print(f"The intercept: {model.intercept_}")

    y_prediction = model.predict(x_test)

    # score the model here:
    score = model.score(x_test, y_test)
    print(f"Model Score: {score}")

    # create the scatter plot here:
    plt.scatter(y_test, y_prediction)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


# perform single feature linear regressions here:
regression_model(
    df[["BreakPointsOpportunities"]],
    df[["Winnings"]],
    "Break Points Opportunities vs Winnings",
    "True Values",
    "Predictions",
)
regression_model(
    df[["ReturnPointsWon"]],
    df[["Winnings"]],
    "Return Points Won vs Winnings",
    "True Values",
    "Predictions",
)
regression_model(
    df[["FirstServeReturnPointsWon"]],
    df[["Winnings"]],
    "First Serve Return Points Won vs Winnings",
    "True Values",
    "predictions",
)

# breakpointsopportunities seems to have the highest model score and strongest correlation with winnings.
# The coefficient: [1859.29482076]
# The intercept: [43478.54034898]
# Model Score: 0.8250746124263673


## perform two feature linear regressions here:


def regression_model_with_multiple_features(features, output, title, x_label, y_label):
    # split the data here:
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2)

    # create and fit the model here:
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f"The coefficient: {model.coef_[0]}")
    print(f"The intercept: {model.intercept_}")

    y_prediction = model.predict(x_test)

    # score the model here:
    score = model.score(x_test, y_test)
    print(f"Model Score: {score}")

    # create the scatter plot here:
    plt.scatter(y_test, y_prediction)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


# perform two feature linear regressions here:
regression_model_with_multiple_features(
    df[["BreakPointsOpportunities", "ReturnPointsWon"]],
    df[["Winnings"]],
    "Break Points Opportunities & Return Points Won vs Winnings",
    "True Values",
    "Predictions",
)
regression_model_with_multiple_features(
    df[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]],
    df[["Winnings"]],
    "Break Points Opportunities & First Serve Return Points Won vs Winnings",
    "True Values",
    "Predictions",
)


## perform multiple feature linear regressions here:

features_set = [
    "BreakPointsOpportunities",
    "ReturnPointsWon",
    "FirstServeReturnPointsWon",
    "TotalPointsWon",
    "Aces",
    "DoubleFaults",
    "FirstServePointsWon",
    "SecondServePointsWon",
]

regression_model_with_multiple_features(
    df[["BreakPointsOpportunities", "ReturnPointsWon", "FirstServeReturnPointsWon"]],
    df[["Winnings"]],
    "Break Points Opportunities, Return Points Won & First Serve Return Points Won vs Winnings",
    "True Values",
    "Predictions",
)
regression_model_with_multiple_features(
    df[
        [
            "BreakPointsOpportunities",
            "ReturnPointsWon",
            "FirstServeReturnPointsWon",
            "TotalPointsWon",
        ]
    ],
    df[["Winnings"]],
    "Break Points Opportunities, Return Points Won, First Serve Return Points Won & Total Points Won vs Winnings",
    "True Values",
    "Predictions",
)
regression_model_with_multiple_features(
    df[features_set],
    df[["Winnings"]],
    "All Features vs Winnings",
    "True Values",
    "Predictions",
)
