/*---------------------------------------------------------------------------*\
Application
    trasportoFoam

Description
    Il solutore piu' semplice che si possa scrivere: trasporto (convezione e
    diffusione) di uno scalare passivo T in un campo di velocita' U letto da file
    e tenuto fisso,

        dT/dt + div(phi T) - div(DT grad T) = 0,

    con phi = U_f . S_f il flusso di faccia e DT la diffusivita' (physicalProperties).
    Stazionario con ddtSchemes steadyState, transitorio con Euler/backward/CrankNicolson.
    Equivale alla function object scalarTransport, ma qui ogni riga e' visibile.
\*---------------------------------------------------------------------------*/

#include "argList.H"           // gli argomenti di riga di comando (-case, -parallel, ...)
#include "simpleControl.H"     // il ciclo esterno con i correttori non ortogonali (porta con se' Time e fvMesh)
#include "fvcFlux.H"           // fvc::flux(U), usato da createPhi.H
#include "fvmDdt.H"            // gli operatori impliciti: derivata temporale,
#include "fvmDiv.H"            //   divergenza (convezione),
#include "fvmLaplacian.H"      //   laplaciano (diffusione)

using namespace Foam;          // nelle applicazioni si scrive Info, fvm::, volScalarField senza il prefisso Foam::

int main(int argc, char *argv[])
{
    #include "setRootCase.H"   // legge gli argomenti (-case, -parallel, ...)
    #include "createTime.H"    // crea l'oggetto runTime da system/controlDict
    #include "createMesh.H"    // legge constant/polyMesh e crea la fvMesh

    simpleControl simple(mesh);

    #include "createFields.H"  // T, U, phi, DT (file accanto a questo)

    Info<< "\nTrasporto dello scalare T nel campo di velocita' assegnato\n" << endl;

    while (simple.loop(runTime))               // avanza il tempo (o l'iterazione) e controlla l'arresto
    {
        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        while (simple.correctNonOrthogonal())  // ripete per nNonOrthogonalCorrectors + 1 volte
        {
            fvScalarMatrix TEqn                // l'equazione, termine per termine: ogni fvm:: legge fvSchemes
            (
                fvm::ddt(T)
              + fvm::div(phi, T)
              - fvm::laplacian(DT, T)
            );

            TEqn.relax();                      // rilassamento implicito (fattore in fvSolution, se presente)
            TEqn.solve();                      // il solutore lineare scelto in fvSolution per T
        }

        runTime.write();                       // scrive T agli istanti di controlDict

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}
