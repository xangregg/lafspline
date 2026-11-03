# Fixtures for test/pspline.test.js: P-spline fits by mgcv (and a dense solve in base R) for comparison with
# src/core/pspline.js. Run from the repository root:
#
#     Rscript test/fixtures/pspline-mgcv.R
#
# writes test/fixtures/pspline-mgcv.json. R 4.6.0, mgcv 1.9-4, jsonlite.
#
# The basis is R's own splines::splineDesign on Eilers & Marx's knots (nseg equal segments on [xl, xr], extended by
# deg segments on each side), independent of the JavaScript basis. mgcv gets it through `paraPen`, so the penalty
# lambda * t(D) %*% D is used as given: no penalty rescaling and no centering constraint (a bs = "ps" smooth would
# have both). Every mgcv fit is checked against a dense solve here before it is written.

suppressPackageStartupMessages({
    library(mgcv)
    library(splines)
    library(jsonlite)
})

TOL <- 1e-6  # relative agreement required between mgcv and the dense solve in this script

basis <- function(x, xl, xr, nseg, deg) {
    dx <- (xr - xl) / nseg
    knots <- xl + (seq_len(nseg + 2 * deg + 1) - 1 - deg) * dx
    knots[deg + 1 + nseg] <- xr  # exact, so splineDesign accepts x = xr
    splineDesign(knots, x, ord = deg + 1, outer.ok = FALSE)
}

dense <- function(B, y, w, P, lambda) {
    G <- crossprod(B, w * B)
    A <- G + lambda * P
    Ainv <- solve(A)
    alpha <- Ainv %*% crossprod(B, w * y)
    fitted <- drop(B %*% alpha)
    ed <- sum(diag(Ainv %*% G))
    rss <- sum(w * (y - fitted)^2)
    list(alpha = drop(alpha), fitted = fitted, ed = ed, rss = rss, Ainv = Ainv, G = G)
}

close <- function(a, b, what) {
    err <- max(abs(a - b)) / max(1, max(abs(b)))
    if (err > TOL)
        stop(sprintf("%s: mgcv and dense solve differ by %g", what, err))
}

make_case <- function(name, x, y, w, nseg, deg, d, lambdas, grid) {
    xl <- min(x)
    xr <- max(x)
    B <- basis(x, xl, xr, nseg, deg)
    nb <- ncol(B)
    D <- diff(diag(nb), differences = d)
    P <- crossprod(D)
    Bg <- basis(grid, xl, xr, nseg, deg)
    n <- length(y)
    fits <- lapply(lambdas, function(lambda) {
        m <- gam(y ~ B - 1, data = list(y = y, B = B), weights = w, paraPen = list(B = list(P)), sp = lambda)
        ref <- dense(B, y, w, P, lambda)
        close(coef(m), ref$alpha, paste(name, lambda, "alpha"))
        close(sum(m$edf), ref$ed, paste(name, lambda, "ed"))
        sigma2 <- ref$rss / (n - ref$ed)
        close(m$sig2, sigma2, paste(name, lambda, "sigma2"))
        close(m$Vp, sigma2 * ref$Ainv, paste(name, lambda, "Vp"))
        list(
            lambda = lambda,
            alpha = unname(coef(m)),
            fitted = unname(fitted(m)),
            ed = sum(m$edf),
            rss = ref$rss,
            sigma = sqrt(m$sig2),
            gcv = m$gcv.ubre,
            hat = unname(m$hat),
            seBayes = sqrt(rowSums((Bg %*% m$Vp) * Bg)),
            seSandwich = sqrt(rowSums((Bg %*% m$Ve) * Bg))
        )
    })
    # mgcv's own GCV choice of lambda (Newton on log lambda), for comparison with the grid search plus refinement
    m <- gam(y ~ B - 1, data = list(y = y, B = B), weights = w, paraPen = list(B = list(P)), method = "GCV.Cp")
    list(
        name = name, x = x, y = y, w = w, nseg = nseg, deg = deg, diff = d, grid = grid,
        basisGrid = lapply(seq_len(nrow(Bg)), function(i) Bg[i, ]),
        fits = fits,
        gcvChoice = list(lambda = unname(m$sp), ed = sum(m$edf), gcv = m$gcv.ubre, fitted = unname(fitted(m)))
    )
}

nile <- read.csv("data/nile-flow.csv", comment.char = "#")

set.seed(20261102)
xs <- sort(round(runif(80) * 30) / 3)  # unequal spacing with ties
ys <- sin(xs / 2) + 0.3 * rnorm(length(xs))
ws <- sample(1:3, length(xs), replace = TRUE)

cases <- list(
    make_case("nile", nile$year, nile$flow, rep(1, nrow(nile)), nseg = 20, deg = 3, d = 2,
        lambdas = c(0.1, 10, 1000), grid = seq(1871, 1970, length.out = 23)),
    make_case("ties-d3", xs, ys, rep(1, length(xs)), nseg = 15, deg = 3, d = 3,
        lambdas = c(1, 100), grid = seq(min(xs), max(xs), length.out = 17)),
    make_case("weighted-quadratic-d1", xs, ys, ws, nseg = 12, deg = 2, d = 1,
        lambdas = c(0.5, 50), grid = seq(min(xs), max(xs), length.out = 17))
)

out <- list(
    generated = format(Sys.Date()),
    r = R.version.string,
    mgcv = as.character(packageVersion("mgcv")),
    cases = cases
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = FALSE), "test/fixtures/pspline-mgcv.json")
cat("wrote test/fixtures/pspline-mgcv.json\n")
