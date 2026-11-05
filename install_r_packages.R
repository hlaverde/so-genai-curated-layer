# Installs HonestDiD and every package it depends on, in the exact versions that
# produced the results, from a dated snapshot of Posit Package Manager.
#
#   Rscript install_r_packages.R
#
# The packages go to the first library on R's path; set R_LIBS to choose it.

r_version <- "4.6.1"
snapshot <- "https://packagemanager.posit.co/cran/2026-11-04"

pinned <- c(
  HonestDiD = "0.2.8",
  alabama = "2025.1.0",
  backports = "1.5.1",
  checkmate = "2.3.4",
  clarabel = "0.11.3",
  cli = "3.6.6",
  cpp11 = "0.5.5",
  CVXR = "1.9.3",
  dplyr = "1.2.1",
  ECOSolveR = "0.6.2",
  farver = "2.1.2",
  foreach = "1.5.2",
  generics = "0.1.4",
  ggplot2 = "4.0.3",
  glue = "1.8.1",
  gmp = "0.7-5.1",
  gtable = "0.3.6",
  highs = "1.15.1-2",
  isoband = "0.3.0",
  iterators = "1.0.14",
  labeling = "0.4.3",
  latex2exp = "0.9.8",
  lifecycle = "1.0.5",
  lpSolveAPI = "5.5.2.0-17.15",
  magrittr = "2.0.5",
  matrixStats = "1.5.0",
  mvtnorm = "1.4-2",
  nleqslv = "3.3.7",
  numDeriv = "2016.8-1.1",
  osqp = "1.0.0",
  pillar = "1.11.1",
  pkgconfig = "2.0.3",
  pracma = "2.4.6",
  purrr = "1.2.2",
  qrng = "0.0-11",
  R6 = "2.6.1",
  RColorBrewer = "1.1-3",
  Rcpp = "1.1.2",
  RcppArmadillo = "15.6.2-1",
  RcppEigen = "0.3.4.0.2",
  Rglpk = "0.6-5.1",
  rlang = "1.3.0",
  S7 = "0.2.2",
  scales = "1.4.0",
  scs = "3.3.1",
  slam = "0.1-56",
  spacefillr = "0.4.0",
  tibble = "3.3.1",
  tidyselect = "1.2.1",
  TruncatedNormal = "2.3",
  utf8 = "1.2.6",
  vctrs = "0.7.3",
  viridisLite = "0.4.3",
  withr = "3.0.3"
)

# Recommended packages that ship with R 4.6.1: checked here, not installed.
shipped <- c(codetools = "0.2-20", lattice = "0.22-9", Matrix = "1.7-5")

if (as.character(getRversion()) != r_version) {
  stop("R ", r_version, " is required; this is R ", getRversion(), call. = FALSE)
}

lib <- .libPaths()[1]
if (file.access(lib, 2) != 0) {
  stop("Library ", lib, " is not writable; set R_LIBS to an existing writable folder",
       call. = FALSE)
}

options(timeout = 600)
install.packages(
  names(pinned),
  lib = lib,
  repos = snapshot,
  dependencies = FALSE,
  type = if (.Platform$OS.type == "windows") "binary" else "source"
)

wanted <- c(pinned, shipped)
found <- vapply(names(wanted), function(p) {
  tryCatch(format(packageVersion(p)), error = function(e) "missing")
}, character(1))
same <- vapply(names(wanted), function(p) {
  found[[p]] != "missing" && packageVersion(p) == package_version(wanted[[p]])
}, logical(1))
if (!all(same)) {
  print(data.frame(package = names(wanted), wanted = wanted, found = found)[!same, ],
        row.names = FALSE)
  stop("These packages do not have the required version", call. = FALSE)
}

library(HonestDiD)
cat(length(wanted), "packages at the required versions, in", lib, "\n")
