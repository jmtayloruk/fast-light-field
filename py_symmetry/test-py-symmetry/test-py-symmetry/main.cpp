//
//  main.cpp
//  test-py-symmetry
//
//  Created by Jonathan Taylor on 11/02/2020.
//  Copyright (c) 2020 Jonathan Taylor. All rights reserved.
//
#include <stdio.h>

double GetTime(void);

int main(int argc, const char * argv[])
{
    void *TestMe(void);
    double t1 = GetTime();
    TestMe();
    double t2 = GetTime();
    printf("Took %lf\n", t2-t1);
    
    return 0;
}
