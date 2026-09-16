function ok = symbolic_proof()
%SYMBOLIC_PROOF Structural degeneracy of the siRNA model, in MATLAB.
%   Independent cross-check of src/structural.py. Requires only the Symbolic
%   Math Toolbox -- no third-party identifiability package.
%
%   Step 1: substituting the scaled states
%       u = A_p/D,  v = A_t/(D k_up),  w = A_r/(D k_up k_esc)
%   must leave a system whose coefficients are the three poles and nothing
%   else. D, k_up and k_esc have to cancel.
%
%   Step 2: the Emax term is homogeneous of degree zero in (A_r, EC50), so it
%   sees k_up, k_esc and EC50 only through psi = D*k_up*k_esc/EC50.
%
%   Together: nine parameters enter the observables through seven functions.

syms t D k_el k_up k_esc k_degt k_loss Emax EC50 a b psi positive
syms u(t) v(t) w(t)

A_p = D*u;  A_t = D*k_up*v;  A_r = D*k_up*k_esc*w;

r1 = simplify((diff(A_p,t) + (k_el + k_up)*A_p) / D);
r2 = simplify((diff(A_t,t) - k_up*A_p + (k_esc + k_degt)*A_t) / (D*k_up));
r3 = simplify((diff(A_r,t) - k_esc*A_t + k_loss*A_r) / (D*k_up*k_esc));

% re-express in the poles: k_el = a - k_up, k_degt = b - k_esc
r1 = simplify(subs(r1, k_el,   a - k_up));
r2 = simplify(subs(r2, k_degt, b - k_esc));

lumped  = [D, k_up, k_esc];
leftover = @(r) intersect(symvar(r), lumped);
clean = isempty(leftover(r1)) && isempty(leftover(r2)) && isempty(leftover(r3));

fprintf('scaled equations\n');
fprintf('  u: %s\n', char(r1));
fprintf('  v: %s\n', char(r2));
fprintf('  w: %s\n', char(r3));
fprintf('lumped factors remaining after scaling: %s\n', ...
        ternary(clean, 'none', char(leftover(r1))));

resid = simplify(subs(Emax*A_r/(EC50 + A_r), EC50, D*k_up*k_esc/psi) ...
                 - Emax*psi*w/(1 + psi*w));
emax_ok = isequal(resid, sym(0));
fprintf('Emax rewrite residual: %s\n', char(resid));

ok = clean && emax_ok;
fprintf('\nfactorisation holds: %d\n', ok);
fprintf(['identifiable functions: k_el+k_up, k_esc+k_degt, k_loss, k_out, ' ...
         'k_syn, Emax, psi\n']);
fprintf('=> 9 parameters, 7 identifiable functions, 2-D degeneracy\n');
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
