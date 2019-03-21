#include "structure.h"

#include <assert.h>

Structure::Structure(unsigned int depth, unsigned int heigth, unsigned int width)
    : depth{depth}
    , heigth{heigth}
    , width{width}
{
    assert(depth>0);
    assert(heigth>0);
    assert(width>0);
}
Structure::Structure(unsigned int heigth, unsigned int width)
    : depth{0}
    , heigth{heigth}
    , width{width}
{
    assert(heigth>0);
    assert(width>0);
}
Structure::Structure(unsigned int depth, const Structure& convolutionStructure, const Structure& prevStructure)
    : depth{depth}
    , heigth{prevStructure.heigth+1-convolutionStructure.heigth}
    , width{prevStructure.width+1-convolutionStructure.width} {}
unsigned int Structure::size() const
{
    return depth*heigth*width;
}
Structure Structure::get_flat() const
{
    return {size(),1,1};
}
