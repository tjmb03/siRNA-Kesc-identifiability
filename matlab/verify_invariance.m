function out = verify_invariance(alpha, beta)
%VERIFY_INVARIANCE Integrate the full nonlinear model under the degeneracy
%   group and confirm plasma and protein are pointwise unchanged.
%
%   k_up -> alpha*k_up,  k_el   -> (k_el+k_up) - alpha*k_up
%   k_esc-> beta *k_esc, k_degt -> (k_esc+k_degt) - beta*k_esc
%   EC50 -> alpha*beta*EC50
%
%   ode15s per the project's stiffness convention (the fastest and slowest
%   poles differ by a factor of ~600).
if nargin < 1, alpha = 1.15; end
if nargin < 2, beta  = 4.00; end

p = reference_params();
a = p.k_el + p.k_up;  b = p.k_esc + p.k_degt;

q = p;
q.k_up   = alpha * p.k_up;    q.k_el   = a - q.k_up;
q.k_esc  = beta  * p.k_esc;   q.k_degt = b - q.k_esc;
q.EC50   = alpha * beta * p.EC50;
assert(q.k_el > 0 && q.k_degt > 0, ...
       'alpha must be < %.3f and beta < %.3f', a/p.k_up, b/p.k_esc);

t    = linspace(0, 180, 1200);
opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-14);
ic   = @(s) [s.dose; 0; 0; s.k_syn/s.k_out];
[~, Y0] = ode15s(@(tt,yy) sirna_rhs(tt,yy,p), t, ic(p), opts);
[~, Y1] = ode15s(@(tt,yy) sirna_rhs(tt,yy,q), t, ic(q), opts);

out.max_difference_protein = max(abs(Y1(:,4)/(q.k_syn/q.k_out) ...
                                   - Y0(:,4)/(p.k_syn/p.k_out)));
out.max_difference_plasma  = max(abs(Y1(:,1) - Y0(:,1))) / p.dose;
out.k_esc_ratio            = q.k_esc / p.k_esc;
out.escape_fraction        = [p.k_esc/(p.k_esc+p.k_degt), ...
                              q.k_esc/(q.k_esc+q.k_degt)];

fprintf('k_esc  %.4f -> %.4f  (%.1fx)\n', p.k_esc, q.k_esc, out.k_esc_ratio);
fprintf('escape fraction  %.1f%% -> %.1f%%\n', 100*out.escape_fraction);
fprintf('max protein difference  %.3e\n', out.max_difference_protein);
fprintf('max plasma  difference  %.3e\n', out.max_difference_plasma);
end
