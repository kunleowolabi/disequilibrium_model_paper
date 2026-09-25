# ============================================================================
# Model 1 cross-validation against the `markets` R package -- CORRECTED
# ============================================================================
# Fix: deviance() is not defined for market_fit S4 objects (confirmed
# twice now). Extracts log-likelihood by parsing summary(fit)'s printed
# text instead, which has worked reliably throughout this validation.
#
# Usage:
#   Rscript validate_model1_markets_package.R
# ============================================================================

if (!requireNamespace("markets", quietly = TRUE)) {
  install.packages("markets", repos = "https://cloud.r-project.org")
}
library(markets)
library(dplyr)

# ---- helper: extract log-likelihood from summary(fit)'s printed output ----
get_logL <- function(fit) {
  s <- capture.output(print(summary(fit)))
  line <- grep("log L", s, value = TRUE)
  if (length(line) == 0) return(NA)
  val <- as.numeric(gsub("[^0-9.-]", "", sub(".*:", "", line[1])))
  return(-val / 2)  # printed as "-2 log L: X" -> logL = -X/2
}

# ---- 1. Load data (v4, corrected 154-obs sample) ----

estimation_sample <- read.csv("estimation_sample_v4_for_R.csv", stringsAsFactors = FALSE)
estimation_sample <- estimation_sample %>%
  mutate(
    angle = 2 * pi * quarter / 4,
    cos_q = cos(angle),
    sin_q = sin(angle)
  )

lagos_discos <- c("Eko", "Ikeja")

df <- estimation_sample %>%
  mutate(
    log_igr = log(igr),
    lagos_group = as.integer(disco %in% lagos_discos),
    log_igr_x_lagos = log_igr * lagos_group,
    disco = factor(disco),
    no_price = 1
  )

cat(sprintf("Loaded %d observations, %d DISCOs\n", nrow(df), length(unique(df$disco))))
cat("\nSample check:\n")
print(head(df[, c("disco", "period", "Q", "log_igr", "log_igr_x_lagos", "pcc", "avg_paf")]))

# ---- 2. Baseline single-start fit ----

cat("\n--- Baseline single-start estimation ---\n")
base_fit <- diseq_basic(
  Q | no_price | disco | period ~ log_igr + log_igr_x_lagos + cos_q + sin_q |
    pcc + avg_paf + cos_q + sin_q + disco,
  data = df,
  correlated_shocks = FALSE,
  verbose = 0
)
base_start <- coef(base_fit)
base_logL <- get_logL(base_fit)
cat(sprintf("Baseline logL: %.4f\n", base_logL))
cat("Baseline coefficients:\n")
print(summary(base_fit))

# ---- 3. Multi-start robustness check ----

cat("\n============================================================\n")
cat("MULTI-START ROBUSTNESS CHECK\n")
cat("============================================================\n")

set.seed(11)
n_starts <- 10
best_fit <- base_fit
best_logL <- base_logL
results_log <- data.frame(start_id = 0, logL = base_logL)
cat(sprintf("  Start  0 (baseline): logL = %.4f\n", base_logL))

for (i in 1:n_starts) {
  jittered_start <- base_start * (1 + rnorm(length(base_start), 0, 0.3))
  names(jittered_start) <- names(base_start)

  fit_i <- tryCatch({
    diseq_basic(
      Q | no_price | disco | period ~ log_igr + log_igr_x_lagos + cos_q + sin_q |
        pcc + avg_paf + cos_q + sin_q + disco,
      data = df,
      correlated_shocks = FALSE,
      verbose = 0,
      estimation_options = list(start = jittered_start, control = list(maxit = 5000))
    )
  }, error = function(e) {
    cat(sprintf("  Start %2d: FAILED -- %s\n", i, conditionMessage(e)))
    NULL
  })

  if (!is.null(fit_i)) {
    logL_i <- get_logL(fit_i)
    results_log <- rbind(results_log, data.frame(start_id = i, logL = logL_i))
    cat(sprintf("  Start %2d: logL = %.4f\n", i, logL_i))
    if (!is.na(logL_i) && logL_i > best_logL) {
      best_logL <- logL_i
      best_fit <- fit_i
      cat("           ^ new best\n")
    }
  } else {
    results_log <- rbind(results_log, data.frame(start_id = i, logL = NA))
  }
}

cat("\n============================================================\n")
cat("MULTI-START SUMMARY\n")
cat("============================================================\n")
print(results_log)
cat(sprintf("\nBest logL found: %.4f\n", best_logL))
cat(sprintf("Our Python result (154 obs, corrected): logL = -621.9556\n"))

if (best_logL > -621.9556) {
  cat("\n>>> R's best exceeds our Python result -- inspect its coefficients closely.\n")
} else {
  cat("\n>>> Our Python result remains the higher-likelihood (better) fit.\n")
}

cat("\n--- Best-fit coefficients ---\n")
print(summary(best_fit))

sink("markets_validation_output_v4.txt")
cat("=== Baseline ===\n"); print(summary(base_fit))
cat("\n=== Multi-start log-likelihoods ===\n"); print(results_log)
cat("\n=== Best fit ===\n"); print(summary(best_fit))
sink()
cat("\nSaved markets_validation_output_v4.txt\n")