/*---------------------------------------------------------------------------*\
Class
    Foam::profiloParabolicoFvPatchVectorField

Description
    Condizione al contorno di velocita' con profilo parabolico (Poiseuille) su un
    patch circolare o su un canale piano:

        U = 2 Um [1 - (r/R)^2] * direzione * f(t)

    con r la distanza del centro della faccia dall'asse (punto 'centro', versore
    'direzione'), Um la velocita' media, R il raggio e f(t) una Function1 opzionale
    del tempo (predefinita: costante 1). Derivata da fixedValue: fornisce al
    solutore i quattro coefficienti di una condizione di Dirichlet.

Usage
    ingresso
    {
        type        profiloParabolico;
        Um          0.1;
        R           0.01;
        centro      (0 0 0);
        direzione   (1 0 0);
        fattore     constant 1;     // oppure: sine { amplitude 0.5; frequency 2; level 1; }
        value       uniform (0.1 0 0);
    }

SourceFiles
    profiloParabolicoFvPatchVectorField.C
\*---------------------------------------------------------------------------*/

#ifndef profiloParabolicoFvPatchVectorField_H
#define profiloParabolicoFvPatchVectorField_H

#include "fixedValueFvPatchFields.H"
#include "Function1.H"

namespace Foam
{

class profiloParabolicoFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private Data

        //- Velocita' media
        scalar Um_;

        //- Raggio del condotto (o semialtezza del canale)
        scalar R_;

        //- Un punto dell'asse
        vector centro_;

        //- Versore dell'asse (verso della velocita')
        vector direzione_;

        //- Fattore moltiplicativo funzione del tempo
        autoPtr<Function1<scalar>> fattore_;


public:

    //- Runtime type information: il nome con cui la condizione e' scelta nel dizionario
    TypeName("profiloParabolico");


    // Constructors

        //- Construct from patch, internal field and dictionary
        profiloParabolicoFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given field onto a new patch
        profiloParabolicoFvPatchVectorField
        (
            const profiloParabolicoFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fieldMapper&
        );

        //- Disallow copy without setting internal field reference
        profiloParabolicoFvPatchVectorField
        (
            const profiloParabolicoFvPatchVectorField&
        ) = delete;

        //- Copy constructor setting internal field reference
        profiloParabolicoFvPatchVectorField
        (
            const profiloParabolicoFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new profiloParabolicoFvPatchVectorField(*this, iF)
            );
        }


    // Member Functions

        //- Aggiorna il valore sul patch (chiamata a ogni valutazione)
        virtual void updateCoeffs();

        //- Scrive la condizione nel file del campo
        virtual void write(Ostream&) const;
};

} // End namespace Foam

#endif
