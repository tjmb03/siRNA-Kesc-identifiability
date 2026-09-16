function model = genssi_sirna()
%GENSSI_SIRNA Model definition for GenSSI 2.0 (Ligon et al., Bioinformatics 2018).
%
%   UNTESTED. This file was written without access to a MATLAB licence, so the
%   field names and the entry-point call below were not executed. Check them
%   against the README of your installed GenSSI version before trusting the
%   output. symbolic_proof.m and sensitivity_rank.m in this folder need only
%   core MATLAB plus the Symbolic Math Toolbox and are the safer cross-check.
%
%   Intended use:
%       genssiMain('genssi_sirna', 6)
%
%   Expected result, from src/structural.py: the model is NOT structurally
%   globally identifiable. The identifiable functions are
%       k_el + k_up,  k_esc + k_degt,  k_loss,  k_out,  k_syn,  Emax,
%       psi = D*k_up*k_esc/EC50
%   giving a two-dimensional degeneracy that contains k_esc. Adding A_t as a
%   third output should reduce it to one dimension.

syms k_el k_up k_esc k_degt k_loss k_out k_syn Emax EC50 D
syms A_p A_t A_r P

model.sym.x = [A_p; A_t; A_r; P];
model.sym.p = [k_el; k_up; k_esc; k_degt; k_loss; k_out; k_syn; Emax; EC50];
model.sym.g = [];                        % no time-varying input; dose is an IC
model.sym.x0 = [D; 0; 0; k_syn/k_out];

model.sym.xdot = [ -(k_el + k_up)*A_p
                    k_up*A_p - (k_esc + k_degt)*A_t
                    k_esc*A_t - k_loss*A_r
                    k_syn*(1 - Emax*A_r/(EC50 + A_r)) - k_out*P ];

% observed: plasma and target protein. Uncomment A_t to add the tissue readout.
model.sym.y = [A_p; P];
% model.sym.y = [A_p; A_t; P];
end
