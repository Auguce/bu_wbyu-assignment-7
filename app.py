from flask import Flask, render_template, request, url_for, session
from flask_session import Session
import numpy as np
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management
app.config["SESSION_TYPE"] = "filesystem"  # Use filesystem for session storage
Session(app)


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # Generate random dataset Y using the specified beta0, beta1, mu, and sigma2
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, alpha=0.5, label="Data points")
    plt.plot(X, slope * X + intercept, color="red", label="Fitted line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Generated Data with Fitted Line")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.7, color='blue', edgecolor='black', label="Simulated Slopes")
    plt.axvline(slope, color='red', linestyle='--', label=f"Observed Slope: {slope:.4f}")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.title("Histogram of Simulated Slopes")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    slopes = np.array(session.get("slopes"))
    intercepts = np.array(session.get("intercepts"))
    slope = session.get("slope")
    intercept = session.get("intercept")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = slopes
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = intercepts
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    fun_message = "Rare event detected!" if p_value <= 0.0001 else None

    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.7, color='blue', edgecolor='black', label="Simulated Statistics")
    plt.axvline(observed_stat, color='red', linestyle='--', label=f"Observed: {observed_stat:.4f}")
    plt.axvline(hypothesized_value, color='green', linestyle='-', label=f"Hypothesis: {hypothesized_value}")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Hypothesis Test for {parameter.capitalize()}")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    slopes = np.array(session.get("slopes"))
    intercepts = np.array(session.get("intercepts"))
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    if parameter == "slope":
        estimates = slopes
        observed_stat = session.get("slope")
        true_param = beta1
    else:
        estimates = intercepts
        observed_stat = session.get("intercept")
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, len(estimates) - 1)
    ci_lower = mean_estimate - t_value * std_estimate
    ci_upper = mean_estimate + t_value * std_estimate
    includes_true = ci_lower <= true_param <= ci_upper

    plt.figure()
    plt.scatter(estimates, np.zeros_like(estimates), alpha=0.5, color="gray")
    plt.axvline(mean_estimate, color='blue', label=f"Mean Estimate: {mean_estimate:.4f}")
    plt.axvline(ci_lower, color='green', linestyle="--", label=f"{confidence_level*100}% CI")
    plt.axvline(ci_upper, color='green', linestyle="--")
    plt.axvline(true_param, color='red', linestyle="-", label=f"True {parameter.capitalize()}: {true_param}")
    plt.title(f"{confidence_level*100}% Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level * 100,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true
    )


if __name__ == "__main__":
    app.run(debug=True)
