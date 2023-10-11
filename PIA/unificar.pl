unifica(X, Y) :- X = Y.
unifica(X, Y) :- compound(X), compound(Y), functor(X, F, Af), functor(Y, G, Ag), F = G, Af = Ag, unifica_args(X, Y).

unifica_args(X, Y) :- X =.. [_ | Xs], Y =.. [_ | Ys], son_iguales(Xs, Ys).
unifica_args(X, X) :- atomic(X).

son_iguales([], []).
son_iguales([X|Xs], [Y|Ys]) :- unifica(X, Y), son_iguales(Xs, Ys).


% Predicado para generar subconjuntos de una lista
tiene_subconjunto([], []).
tiene_subconjunto([X|T], [X|Subconjunto]) :-
    tiene_subconjunto(T, Subconjunto).
tiene_subconjunto([_|T], Subconjunto) :-
    tiene_subconjunto(T, Subconjunto).