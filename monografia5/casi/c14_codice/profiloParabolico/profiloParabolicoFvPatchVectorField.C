#include "profiloParabolicoFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::profiloParabolicoFvPatchVectorField::profiloParabolicoFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    Um_(dict.lookup<scalar>("Um")),
    R_(dict.lookup<scalar>("R")),
    centro_(dict.lookupOrDefault<vector>("centro", vector::zero)),
    direzione_(normalised(dict.lookupOrDefault<vector>("direzione", vector(1, 0, 0)))),
    fattore_
    (
        dict.found("fattore")
      ? Function1<scalar>::New("fattore", db().time().userUnits(), dimless, dict)
      : autoPtr<Function1<scalar>>(nullptr)
    )
{
    // valore iniziale: quello del profilo (se il campo e' gia' disponibile) o 'value' se presente
    if (dict.found("value"))
    {
        fvPatchVectorField::operator=(vectorField("value", iF.dimensions(), dict, p.size()));
    }
    else
    {
        updateCoeffs();
    }
}


Foam::profiloParabolicoFvPatchVectorField::profiloParabolicoFvPatchVectorField
(
    const profiloParabolicoFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    Um_(ptf.Um_),
    R_(ptf.R_),
    centro_(ptf.centro_),
    direzione_(ptf.direzione_),
    fattore_(ptf.fattore_, false)
{}


Foam::profiloParabolicoFvPatchVectorField::profiloParabolicoFvPatchVectorField
(
    const profiloParabolicoFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    Um_(ptf.Um_),
    R_(ptf.R_),
    centro_(ptf.centro_),
    direzione_(ptf.direzione_),
    fattore_(ptf.fattore_, false)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::profiloParabolicoFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // distanza di ogni centro di faccia dall'asse: componente di (Cf - centro) ortogonale alla direzione
    const vectorField d(patch().Cf() - centro_);
    const scalarField r(mag(d - (d & direzione_)*direzione_));

    scalar f = 1;
    if (fattore_.valid())
    {
        f = fattore_->value(db().time().value());
    }

    // il profilo di Poiseuille: 2 Um (1 - r^2/R^2), mai negativo
    fixedValueFvPatchVectorField::operator==
    (
        direzione_*(2*Um_*f*max(1 - sqr(r/R_), scalar(0)))
    );

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::profiloParabolicoFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    writeEntry(os, "Um", Um_);
    writeEntry(os, "R", R_);
    writeEntry(os, "centro", centro_);
    writeEntry(os, "direzione", direzione_);
    if (fattore_.valid())
    {
        writeEntry(os, db().time().userUnits(), dimless, fattore_());
    }
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    // registra la classe: da qui in poi 'type profiloParabolico' e' riconosciuto per i campi vettoriali
    makePatchTypeField
    (
        fvPatchVectorField,
        profiloParabolicoFvPatchVectorField
    );
}
