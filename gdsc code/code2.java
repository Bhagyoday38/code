public class SumOfNumbers 
{
    public static void main(String[] args) 
{
        int sum = 0;
        int previousNumber = 0;

        for (int i = 1; i <= 10; i++) 
{
            sum += i;
            System.out.println("The sum of the first " + i + " numbers is " + sum);
            previousNumber = i;
        }
    }
}
