function dy = sirna_rhs(~, y, p)
%SIRNA_RHS Four-compartment siRNA PK/PD model.
%   States: [A_p plasma; A_t tissue/endosomal; A_r RISC-loaded; P protein]
%   The Emax term multiplies a synthesis rate: indirect response, not
%   Michaelis-Menten kinetics.
A_p = y(1); A_t = y(2); A_r = y(3); P = y(4);
inhib = p.Emax * A_r / (p.EC50 + A_r);
dy = [ -(p.k_el + p.k_up) * A_p
        p.k_up * A_p - (p.k_esc + p.k_degt) * A_t
        p.k_esc * A_t - p.k_loss * A_r
        p.k_syn * (1 - inhib) - p.k_out * P ];
end
