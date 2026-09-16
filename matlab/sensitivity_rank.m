function out = sensitivity_rank()
%SENSITIVITY_RANK Rank of the relative sensitivity map under three designs.
%   Reproduces identifiability_ladder() in src/structural.py. Expected:
%   2 -> 1 -> 0 unresolved directions.
%
%   Valid for finite-difference steps h in roughly [1e-6, 1e-4].
%
%   Each readout is sampled only where it is quantifiable. Plasma past ~1 day
%   has underflowed to numerical zero and relative sensitivities computed there
%   are integrator noise that swamps the SVD.
p     = reference_params();
names = {'k_el','k_up','k_esc','k_degt','k_loss','k_out','k_syn','Emax','EC50'};

tp = linspace(0.02, 1.0, 25);                       % plasma
tt = linspace(0.05, 7.0, 25);                       % tissue
tq = [linspace(0.25, 7.0, 20), linspace(10, 180, 30)];  % protein

designs = { struct('label','plasma + protein', ...
                   'obs',{{1,4}}, 'times',{{tp,tq}}, 'free',{names}), ...
            struct('label','plasma + protein + total tissue siRNA', ...
                   'obs',{{1,2,4}}, 'times',{{tp,tt,tq}}, 'free',{names}), ...
            struct('label','the above, with k_degt fixed by a stability assay', ...
                   'obs',{{1,2,4}}, 'times',{{tp,tt,tq}}, ...
                   'free',{setdiff(names,{'k_degt'},'stable')}) };

out = struct('design',{},'n_free',{},'rank',{},'unresolved',{},'sv',{});
for d = 1:numel(designs)
    D = designs{d};
    S = sensitivity_matrix(p, D.obs, D.times, D.free);
    sv = svd(S);  sv = sv / sv(1);
    % Rank at the largest gap, not at a fixed cutoff: the smallest singular
    % value moves by two orders of magnitude with the finite-difference step,
    % while the gap sits after the same index. A gap only counts as a rank
    % deficiency if it is large -- a full-rank matrix still has a largest
    % ratio (about 4 here) and reading that as a null space invents
    % degeneracies that are not there.
    ratios = sv(1:end-1) ./ sv(2:end);
    [gap, k] = max(ratios);
    if gap > 1e3, r = k; else, r = numel(sv); end
    out(d).design     = D.label;
    out(d).n_free     = numel(D.free);
    out(d).rank       = r;
    out(d).unresolved = numel(D.free) - r;
    out(d).sv         = sv(:)';
    out(d).gap        = gap;
    fprintf('%-52s free=%d rank=%d unresolved=%d\n', ...
            D.label, out(d).n_free, r, out(d).unresolved);
    if out(d).unresolved > 0
        [~,~,V] = svd(S);
        for k = r+1:numel(D.free)
            v = V(:,k);  keep = abs(v) > 0.05;
            fprintf('      null: ');
            fprintf('%s %+.3f  ', [D.free(keep); num2cell(v(keep)')]{:});
            fprintf('\n');
        end
    end
end
end

function S = sensitivity_matrix(p, obs, times, free)
h = 1e-5;
base = observe(p, obs, times);
S = zeros(numel(base), numel(free));
for j = 1:numel(free)
    f  = free{j};
    up = p; up.(f) = p.(f) * (1 + h);
    dn = p; dn.(f) = p.(f) * (1 - h);
    S(:,j) = (observe(up,obs,times) - observe(dn,obs,times)) ./ (2*h) ./ base;
end
end

function v = observe(p, obs, times)
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-16);
v = [];
for k = 1:numel(obs)
    t = times{k};
    [~, Y] = ode15s(@(tt,yy) sirna_rhs(tt,yy,p), [0 t], ...
                    [p.dose; 0; 0; p.k_syn/p.k_out], opts);
    v = [v; Y(2:end, obs{k})];  %#ok<AGROW>
end
end
