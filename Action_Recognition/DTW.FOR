      SUBROUTINE DTW(ADIM1,BDIM1,ADIM2,BDIM2,A,B,ABDIST)
C
C DTW (Dynamic Time Warping)
C Version: 1.0
C Created: 05/11/2019
C
C Copyright (C) 2019 Orange – All rights reserved
C
C MIT License
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to deal
C in the Software without restriction, including without limitation the rights
C to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
C copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in all
C copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
C OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
C SOFTWARE.
C 
C Documentation of the algorithm in
C https://en.wikipedia.org/wiki/Dynamic_time_warping
C
C FOR PYTHON2 COMPILE WITH: f2py -m dtw -c DTW.FOR
C FOR PYTHON3 COMPILE WITH: f2py3 -m dtw -c DTW.FOR
C
      INTEGER ADIM1,BDIM1,ADIM2,BDIM2
      REAL A(ADIM2,ADIM1),B(BDIM2,BDIM1)
      REAL ABDIST(ADIM2,BDIM2)
      REAL TAB(ADIM1+1,BDIM1+1)
      REAL UPPERBOUND, INTERMIN, COST
Cf2py intent(out) ABDIST
      DO 50 K=1,ADIM2
        DO 40 L=1,BDIM2
C
C         Compares sequence A(K,.) to sequence B(L,.)
          TAB(1,1) = 0
C         Computes the upperbound
          UPPERBOUND = 0
          DO 5 I=1,ADIM1
            UPPERBOUND = UPPERBOUND + ABS(A(K,I))
    5     CONTINUE 
          DO 8 J=1,BDIM1
            UPPERBOUND = UPPERBOUND + ABS(B(L,J))
    8     CONTINUE 
C         Initialize the TAB array to UPPERBOUND
          DO 10 I=2,ADIM1+1
            TAB(I,1) = UPPERBOUND
   10     CONTINUE
          DO 15 J=2,BDIM1+1
            TAB(1,J) = UPPERBOUND
   15     CONTINUE
C         Perform the comparison
          DO 30 I=2,ADIM1+1
            DO 20 J=2,BDIM1+1
               COST = ABS(A(K,I-1)-B(L,J-1))
               INTERMIN = MIN(TAB(I-1,J),TAB(I,J-1))
               TAB(I,J) = COST + MIN(INTERMIN,TAB(I-1,J-1))
   20       CONTINUE
   30     CONTINUE
          ABDIST(K,L) = TAB(ADIM1+1,BDIM1+1)
C
   40   CONTINUE
   50 CONTINUE
      RETURN
      END


