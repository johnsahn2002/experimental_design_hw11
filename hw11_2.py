import numpy as np
import statsmodels.api as sm
from scipy import stats

# Define unscaled AR(1) error generator
def make_error_unscaled(corr_const, num):
    err = []
    prev = np.random.normal(0, 1)
    for _ in range(num):
        prev = corr_const * prev + np.random.normal(0, 1)
        err.append(prev)
    return np.array(err)

# Main analysis function
def question4_unscaled_analysis():
    np.random.seed(0)
    num = 1000
    true_event_time = int(num / 2)
    
    # Generate market returns
    R_market = np.random.normal(0, 1, num) + np.arange(num) / num
    
    # Generate AR(1) error and apply to R_target
    autocorr_error = make_error_unscaled(0.9, num)
    R_target = 2 + R_market + autocorr_error + (np.arange(num) == true_event_time + 1) * 2
    
    false_positives = 0
    valid_tests = 0
    alpha_level = 0.05

    # Run placebo tests on all fictitious event times (excluding the true one)
    for fictitious_event_time in range(50, num - 50):
        if fictitious_event_time == true_event_time:
            continue

        # Fit regression model before the fictitious event
        results = sm.OLS(R_target[:fictitious_event_time],
                         sm.add_constant(R_market[:fictitious_event_time])).fit()
        resid = R_target - results.predict(sm.add_constant(R_market))

        # Compute t-statistic for fictitious event + 1
        if fictitious_event_time + 1 < num:
            t_stat = resid[fictitious_event_time + 1] / resid[:fictitious_event_time].std(ddof=2)
            if abs(t_stat) > stats.t.ppf(1 - alpha_level / 2, fictitious_event_time - 2):
                false_positives += 1
            valid_tests += 1

    # Final result
    false_positive_rate = false_positives / valid_tests
    print(f"False positive rate (unscaled AR(1) error): {false_positive_rate:.3f}")
    return false_positive_rate

# Run it
question4_unscaled_analysis()
