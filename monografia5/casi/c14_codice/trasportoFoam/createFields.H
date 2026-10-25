// --- i campi del solutore: letti da 0/ (MUST_READ) o calcolati

Info<< "Lettura del campo T\n" << endl;
volScalarField T
(
    IOobject
    (
        "T",
        runTime.name(),           // la cartella temporale corrente
        mesh,
        IOobject::MUST_READ,      // deve esistere in 0/
        IOobject::AUTO_WRITE      // viene scritto a ogni istante di scrittura
    ),
    mesh
);

Info<< "Lettura del campo U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.name(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Lettura della diffusivita' DT da physicalProperties\n" << endl;
IOdictionary physicalProperties
(
    IOobject
    (
        "physicalProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);
dimensionedScalar DT("DT", dimKinematicViscosity, physicalProperties);   // [m^2/s]: il controllo dimensionale e' automatico

// il flusso di faccia phi = U_f . S_f: letto se esiste (es. da un calcolo precedente), altrimenti calcolato
#include "createPhi.H"
