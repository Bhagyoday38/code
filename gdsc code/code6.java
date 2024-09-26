import java.util.Scanner;

public class DivisibleByFive 
{

    public static void main(String[] args) 
{
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the size of the list: ");
        int size = scanner.nextInt();
        System.out.println("Enter the elements of the list: ");
        int[] list = new int[size];
        for (int i = 0; i < size; i++)
 {
            list[i] = scanner.nextInt();
        }
        for (int number : list) 
{
            if (number % 5 == 0) 
{
                System.out.println(number);
            }
        }
    }
}
