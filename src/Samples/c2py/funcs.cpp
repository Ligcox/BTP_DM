int add(int i, int j) {
    for (int i = 0; i < 10000; i++)
    {
        j += i;
    }
    
    return i + j;
};