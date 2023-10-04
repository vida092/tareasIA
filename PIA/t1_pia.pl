progenitor(pam,bob).
progenitor(tom,bob).
progenitor(tom,liz).
progenitor(bob,ann).
progenitor(bob,pat).
progenitor(pat,jim).

mujer(pam).
mujer(liz).
mujer(pat).
mujer(ann).
hombre(tom).
hombre(bob).
hombre(jim).
%Definimos las relaciones hermana y hermano
hermana(X,Y) :-
    progenitor(Z,X),
    progenitor(Z,Y),
    mujer(X),
    dif(X,Y).
hermano(X,Y) :-
    progenitor(Z,X),
    progenitor(Z,Y),
    hombre(X),
    dif(X,Y).

ancestro(X,Z) :-
    progenitor(X,Z).

ancestro(X,Z) :-
    progenitor(X,Y),
    ancestro(Y,Z).

tia(X,Y):-
    progenitor(Z,Y),
    hermana(X,Z).

tio(X,Y):-
    progenitor(Z,Y),
    hermano(X,Z).

sobrine(X,Z) :-
    (tia(Z, X) ; tio(Z, X)).


animal(leon).
animal(tigre).
mamifero(leon).