#ifndef STRUCTURE_H_INCLUDED
#define STRUCTURE_H_INCLUDED

class Structure
{
public:
    unsigned int depth;
    unsigned int heigth;
    unsigned int width;
    Structure(unsigned int depth, unsigned int heigth, unsigned int width);
    Structure(unsigned int heigth, unsigned int width);
    Structure(unsigned int depth, const Structure& convolutionStructure, const Structure& prevStructure);
    unsigned int size() const;
    Structure get_flat() const;
};

#endif // STRUCTURE_H_INCLUDED
