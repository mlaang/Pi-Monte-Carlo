
/* Generates pseudo-random floating points numbers in [0,1). */
float random_uniform(unsigned int state[static const 5]) {
    //The state contains the current state of a pseudo-random number generator
	//this operation below uses a variant called XORWOW
    unsigned int s,
	             t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	state[4] += 362437;

	//t + state[4] should now contain 32 bits of pseudorandomness.
	//It's not unusual to compute something like (float)(t + state[4])/)(float)INT_MAX), but this
	//requires division. Another way to create a random number is to use bit operations to turn something
	//directly into a float

	//A 32-bit floating-point number has a sign bit in the highest position. Sice we are to generate a positive number
	//this is to be zero. Then there are eight bits of exponent and 23 bits for the fractional part. We will fill
	//the fractional part with bits of randomness from t + state[4] by shifting it to create zeros in the nine highest
	//bits. We then add the appropriate exponent and sign bits using bitwise or with the hexadecimal representation of 1.0f.
	//This produces a number that has the exponent of 1.0f, but a bunch of decimals instead of, up to something like 1.999.
	//Finally one is subtracted so as to obtain a number in [0,1).

	union {
	    unsigned int ui;
		float f; 
	} ret;
	ret.ui = 0x3F800000 | t + state[4] >> 9;
	return ret.f - 1.0f;
}	

/* Counts points from a certain rectangle that intersect a disc. */
__kernel void count_points(__global int* output, const unsigned int N) {
    const int id = get_global_id(0);
	const float c = 1.0f - sqrt(3.0f)/2.0f,
	            d = sqrt(3.0f)/2.0f;
	int i,
	    a = 0;
	float x, y;
	unsigned int seed[5];
	seed[0] = id;
	for(i = 0; i != N; ++i) {
	    x = random_uniform(seed)*0.5f;
		y = random_uniform(seed)*c;
		if(y < sqrt(1 - x*x) - d)
		    ++a;
	}
	output[id] = a;
}