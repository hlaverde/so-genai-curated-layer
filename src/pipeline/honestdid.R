# HonestDiD (Rambachan and Roth 2023), relative-magnitudes restriction,
# C-LF confidence sets, full cluster-robust event-study VCV.
# Estimand: theta = average of the complete post-period event-study bins.
# A coarse grid, widened until no confidence-set endpoint touches its edge, finds
# the endpoints; a grid of step theta_step around each one locates it.
# usage: Rscript honestdid.R <dir with honestdid_betahat.csv, honestdid_sigma.csv, honestdid_meta.csv>
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
suppressPackageStartupMessages(library(HonestDiD))
b <- read.csv(file.path(d, "honestdid_betahat.csv"))$beta
V <- as.matrix(read.csv(file.path(d, "honestdid_sigma.csv"), header = FALSE))
meta <- read.csv(file.path(d, "honestdid_meta.csv"))
nPre <- meta$n_pre
nPost <- meta$n_post
# target weights: equal by default, or honestdid_lvec.csv when present
lfile <- file.path(d, "honestdid_lvec.csv")
l <- if (file.exists(lfile)) matrix(read.csv(lfile)$l, ncol = 1) else matrix(rep(1 / nPost, nPost), ncol = 1)
post <- (nPre + 1):(nPre + nPost)
theta <- sum(l * b[post])
se <- sqrt(as.numeric(t(l) %*% V[post, post] %*% l))
theta_step <- 4e-5
rm_step <- 0.005
rm_cs <- function(M, glb, gub, n) {
  suppressWarnings(createSensitivityResults_relativeMagnitudes(
    betahat = b, sigma = V, numPrePeriods = nPre, numPostPeriods = nPost,
    Mbarvec = M, l_vec = l, gridPoints = n, grid.lb = glb, grid.ub = gub))
}
# grid of step theta_step on the multiples of theta_step, so it contains zero whenever it reaches it
rm_fine <- function(M, glb, gub) {
  lo <- floor(glb / theta_step)
  hi <- ceiling(gub / theta_step)
  rm_cs(M, lo * theta_step, hi * theta_step, hi - lo + 1)
}
rm_at <- function(M) {
  w <- 0.5
  repeat {
    glb <- theta - w
    gub <- theta + w
    r <- rm_cs(M, glb, gub, 1000)
    step <- (gub - glb) / 999
    edge <- abs(r$lb - glb) < step / 2 || abs(r$ub - gub) < step / 2
    if (!edge) break
    w <- 2 * w
    if (w > 64) stop("confidence set does not close within the grid")
  }
  data.frame(Mbar = M, lb = rm_fine(M, r$lb - step, r$lb + step)$lb, ub = rm_fine(M, r$ub - step, r$ub + step)$ub,
             grid_step = theta_step)
}
rm_res <- do.call(rbind, lapply(c(0, 0.25, 0.5, 0.75, 1, 1.5, 2), rm_at))
write.csv(rm_res, file.path(d, "honestdid_results.csv"), row.names = FALSE)

# Breakdown of Delta-RM: the first Mbar on a grid of step rm_step, between the
# last coarse value that excludes zero and the first that includes it, whose
# confidence set contains zero, tested on the fine theta grid around zero.
# Confidence sets grow with Mbar, so bisection over the grid returns the same
# value as evaluating every grid point. An empty window (no accepted theta
# near zero) gives infinite endpoints and counts as excluding zero.
near_zero <- function(M) {
  r <- rm_fine(M, -0.002, 0.002)
  data.frame(Mbar = M, lb = r$lb, ub = r$ub, contains = r$lb <= theta_step / 2 && r$ub >= -theta_step / 2)
}
inc_rm <- rm_res$lb <= 0 & rm_res$ub >= 0
evals <- list()
if (inc_rm[1]) {
  bd_rm <- data.frame(breakdown = 0, last_excluding = NA, step = rm_step, theta_step = theta_step)   # includes zero at Mbar = 0
} else if (!any(inc_rm)) {
  bd_rm <- data.frame(breakdown = NA, last_excluding = max(rm_res$Mbar), step = rm_step, theta_step = theta_step)  # beyond the grid
} else {
  hi_rm <- min(rm_res$Mbar[inc_rm])
  lo_rm <- max(rm_res$Mbar[!inc_rm & rm_res$Mbar < hi_rm])
  g <- round(seq(lo_rm, hi_rm, length.out = round((hi_rm - lo_rm) / rm_step) + 1), 6)
  i <- 1
  j <- length(g)
  while (j - i > 1) {
    k <- (i + j) %/% 2
    e <- near_zero(g[k])
    evals[[length(evals) + 1]] <- e
    if (e$contains) j <- k else i <- k
  }
  bd_rm <- data.frame(breakdown = g[j], last_excluding = g[i], step = rm_step, theta_step = theta_step)
}
write.csv(bd_rm, file.path(d, "honestdid_rm_breakdown.csv"), row.names = FALSE)
if (length(evals)) write.csv(do.call(rbind, evals), file.path(d, "honestdid_rm_fine.csv"), row.names = FALSE)
# theta and its 95% CI with the t(G-1) reference of the tag-clustered estimates
q <- qt(0.975, meta$df)
write.csv(data.frame(theta = theta, se = se, orig_lb = theta - q * se, orig_ub = theta + q * se,
                     version = as.character(packageVersion("HonestDiD"))),
          file.path(d, "honestdid_original.csv"), row.names = FALSE)

# Smoothness restriction Delta-SD(M): M bounds the change in the slope of the
# differential trend between consecutive 13-week bins (FLCI). The breakdown is
# the first M on a grid of step sd_step whose interval contains zero.
sd_step <- 2e-5
sd_at <- function(Mvec) as.data.frame(createSensitivityResults(
  betahat = b, sigma = V, numPrePeriods = nPre, numPostPeriods = nPost,
  method = "FLCI", Mvec = Mvec, l_vec = l))
sd <- sd_at(c(0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05))
inc <- sd$lb <= 0 & sd$ub >= 0
write.csv(sd[, c("M", "lb", "ub")], file.path(d, "honestdid_sd_results.csv"), row.names = FALSE)
if (inc[1]) {
  bd_sd <- data.frame(breakdown = 0, last_excluding = NA, step = NA)        # includes zero under a linear trend (M = 0)
} else if (!any(inc)) {
  bd_sd <- data.frame(breakdown = NA, last_excluding = max(sd$M), step = NA)       # excludes zero on the whole grid
} else {
  hi <- min(sd$M[inc])
  lo <- max(sd$M[!inc & sd$M < hi])
  g <- round(seq(lo, hi, length.out = round((hi - lo) / sd_step) + 1), 8)
  i <- 1
  j <- length(g)
  while (j - i > 1) {
    k <- (i + j) %/% 2
    e <- sd_at(g[k])
    if (e$lb <= 0 && e$ub >= 0) j <- k else i <- k
  }
  bd_sd <- data.frame(breakdown = g[j], last_excluding = g[i], step = sd_step)
}
write.csv(bd_sd, file.path(d, "honestdid_sd_breakdown.csv"), row.names = FALSE)
